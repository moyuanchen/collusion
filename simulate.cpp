// main.cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <utility>
#include <functional>
#include <string>

// For the inverse CDF (quantile) of a normal distribution,
// you can use Boost.Math (make sure to link with Boost)
#include <boost/math/distributions/normal.hpp>

// --- Utility: Hash for std::pair ---
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1,T2>& p) const {
         auto h1 = std::hash<T1>{}(p.first);
         auto h2 = std::hash<T2>{}(p.second);
         return h1 ^ (h2 << 1);
    }
};

// --- QTable Class ---
// Stores a mapping from (p, v) states to a vector of Q-values.
class QTable {
public:
    int Np, Nv, Nx;
    std::vector<std::pair<int,int>> states;
    std::unordered_map<std::pair<int,int>, std::vector<double>, pair_hash> Q;

    QTable(int Np_, int Nv_, int Nx_)
      : Np(Np_), Nv(Nv_), Nx(Nx_) {
        // Create state space (grid over p and v indices)
        for (int p = 0; p < Np; p++) {
            for (int v = 0; v < Nv; v++) {
                std::pair<int,int> state = {p, v};
                states.push_back(state);
                Q[state] = std::vector<double>(Nx, 0.0);
            }
        }
    }

    double get_Q_value(const std::pair<int,int>& state, int action) {
        return Q[state][action];
    }

    int get_best_action(const std::pair<int,int>& state) {
        auto& qvec = Q[state];
        return static_cast<int>(std::distance(qvec.begin(), std::max_element(qvec.begin(), qvec.end())));
    }

    double get_best_value(const std::pair<int,int>& state) {
        auto& qvec = Q[state];
        return *std::max_element(qvec.begin(), qvec.end());
    }

    void update(const std::pair<int,int>& state, int action, double value) {
        Q[state][action] = value;
    }
};

// --- Fixed Point Solvers ---
// Solve for chi^N and chi^M using fixed-point iteration.
std::pair<double,double> solve_chiN(double I, double xi, double sigma_u, double sigma_v, double theta,
                                    double tol = 1e-12, int max_iter = 10000) {
    double chi = 0.1;
    double lam;
    for (int iter = 0; iter < max_iter; iter++) {
        double gamma = (I * chi) / ((I * chi) * (I * chi) + std::pow(sigma_u / sigma_v, 2));
        lam = (theta * gamma + xi) / (theta + xi * xi);
        double new_chi = 1.0 / ((I + 1) * lam);
        if (std::abs(new_chi - chi) < tol)
            return {new_chi, lam};
        chi = new_chi;
    }
    throw std::runtime_error("solve_chiN did not converge");
}

std::pair<double,double> solve_chiM(double I, double xi, double sigma_u, double sigma_v, double theta,
                                    double tol = 1e-12, int max_iter = 10000) {
    double chi = 0.1;
    double lam;
    for (int iter = 0; iter < max_iter; iter++) {
        double gamma = (I * chi) / ((I * chi) * (I * chi) + std::pow(sigma_u / sigma_v, 2));
        lam = (theta * gamma + xi) / (theta + xi * xi);
        double new_chi = 1.0 / (2.0 * I * lam);
        if (std::abs(new_chi - chi) < tol)
            return {new_chi, lam};
        chi = new_chi;
    }
    throw std::runtime_error("solve_chiM did not converge");
}

// --- InformedAgent Class ---
// Represents an informed trader. Includes a Q-table, state counters, and grid discretization.
class InformedAgent {
public:
    int Np, Nv, n_actions;
    double rho, alpha, beta;
    double sigma_v, v_bar, sigma_u, xi, theta, iota;
    QTable Q;
    std::unordered_map<std::pair<int,int>, int, pair_hash> state_count;
    std::unordered_map<std::pair<int,int>, int, pair_hash> last_optimal;
    int convergence_counter;
    std::vector<double> v_discrete;  // grid for v
    std::vector<double> x_discrete;  // grid for x (order flow)
    std::vector<double> p_discrete;  // grid for p (price)
    double chiN, lambdaN, chiM, lambdaM, x_n, x_m;

    std::mt19937 rng;
    std::uniform_real_distribution<> dist_uniform;

    InformedAgent(int Np_ = 31, int Nv_ = 10, int Nx_ = 15,
                  double rho_ = 0.95, double alpha_ = 0.01, double beta_ = 1e-5,
                  double sigma_v_ = 1, double v_bar_ = 1, double sigma_u_ = 0.1,
                  double xi_ = 500, double theta_ = 0.1, double iota_ = 0.1)
      : Np(Np_), Nv(Nv_), n_actions(Nx_), rho(rho_), alpha(alpha_), beta(beta_),
        sigma_v(sigma_v_), v_bar(v_bar_), sigma_u(sigma_u_), xi(xi_), theta(theta_), iota(iota_),
        Q(Np_, Nv_, Nx_), convergence_counter(0), dist_uniform(0.0, 1.0) {
        rng.seed(std::random_device{}());
        get_discrete_states();
        initialize_Q();
    }

    // Epsilon decay based on state visit counts.
    double get_epsilon(const std::pair<int,int>& state) {
        int count = state_count[state];
        state_count[state] = count + 1;
        return std::exp(-beta * count);
    }

    // Choose an action using an ε-greedy policy.
    int get_action(const std::pair<int,int>& state) {
        double epsilon = get_epsilon(state);
        double r = dist_uniform(rng);
        if (r < epsilon) {
            std::uniform_int_distribution<> dist_action(0, n_actions - 1);
            return dist_action(rng);
        } else {
            int optimal_action = Q.get_best_action(state);
            check_convergence(state, optimal_action);
            return optimal_action;
        }
    }

    // Q-value update rule.
    void update(const std::pair<int,int>& state, int action, double reward, const std::pair<int,int>& next_state) {
        double learning = alpha * (reward + rho * Q.get_best_value(next_state));
        double memory = (1 - alpha) * Q.get_Q_value(state, action);
        double value = learning + memory;
        Q.update(state, action, value);
    }

    // Discretize the value (v) space using the inverse CDF of a standard normal.
    void get_grid_point_values_v() {
        v_discrete.resize(Nv);
        boost::math::normal_distribution<> norm(0.0, 1.0);
        for (int k = 1; k <= Nv; k++) {
            double grid_point = (2.0 * k - 1) / (2.0 * Nv);
            double value = boost::math::quantile(norm, grid_point);
            v_discrete[k - 1] = v_bar + sigma_v * value;
        }
    }

    // Compute grid points for the order flow (x) using the fixed-point solvers.
    void get_grid_point_values_x() {
        // Here I is set to 3 (number of informed traders)
        auto resN = solve_chiN(3, xi, sigma_u, sigma_v, theta);
        auto resM = solve_chiM(3, xi, sigma_u, sigma_v, theta);
        chiN = resN.first; lambdaN = resN.second;
        chiM = resM.first; lambdaM = resM.second;
        x_n = chiN;
        x_m = chiM;
        double span_x = std::abs(x_n - x_m);
        double low = -std::max(x_n, x_m) - iota * span_x;
        double high = std::max(x_n, x_m) + iota * span_x;
        x_discrete.resize(n_actions);
        for (int i = 0; i < n_actions; i++) {
            x_discrete[i] = low + (high - low) * i / (n_actions - 1);
        }
    }

    // Compute grid points for the price (p)
    void get_grid_point_values_p() {
        double lambda_for_p = std::max(lambdaN, lambdaM);
        double ph = v_bar + lambda_for_p * (3 * std::max(x_n, x_m) + sigma_u * 1.96);
        double pl = v_bar - lambda_for_p * (3 * std::max(x_n, x_m) + sigma_u * 1.96);
        double span_p = ph - pl;
        p_discrete.resize(Np);
        for (int i = 0; i < Np; i++) {
            p_discrete[i] = (pl - iota * span_p) + ((ph + iota * span_p) - (pl - iota * span_p)) * i / (Np - 1);
        }
    }

    // Prepare all the discretized grids.
    void get_discrete_states() {
        get_grid_point_values_v();
        get_grid_point_values_x();
        get_grid_point_values_p();
    }

    // Convert a continuous (p,v) pair to discrete grid indices.
    std::pair<int,int> continuous_to_discrete(double p, double v) {
        int p_idx = 0, v_idx = 0;
        double min_diff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < p_discrete.size(); i++) {
            double diff = std::abs(p_discrete[i] - p);
            if(diff < min_diff) { min_diff = diff; p_idx = static_cast<int>(i); }
        }
        min_diff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < v_discrete.size(); i++) {
            double diff = std::abs(v_discrete[i] - v);
            if(diff < min_diff) { min_diff = diff; v_idx = static_cast<int>(i); }
        }
        return {p_idx, v_idx};
    }

    // Initialize the Q-table with a computed value (mimicking the Python code).
    void initialize_Q() {
        for (int p = 0; p < Np; p++) {
            for (int v = 0; v < Nv; v++) {
                std::pair<int,int> state = {p, v};
                for (int x = 0; x < n_actions; x++) {
                    double value = 0.0;
                    for (int x_i = 0; x_i < n_actions; x_i++) {
                        value += v_discrete[v] - (v_bar + lambdaN * (x_n + (3 - 1) * x_discrete[x_i])); // I = 3
                    }
                    value *= x_discrete[x] / ((1 - rho) * n_actions);
                    Q.update(state, x, value);
                }
            }
        }
    }

    // Check for convergence by comparing the current optimal action with the last recorded one.
    void check_convergence(const std::pair<int,int>& state, int action) {
        if (last_optimal.find(state) == last_optimal.end() || last_optimal[state] != action) {
            last_optimal[state] = action;
            convergence_counter = 0;
        } else {
            convergence_counter++;
        }
    }
};

// --- PreferredHabitatAgent Class ---
// A simple agent that selects an action based on a preferred price.
class PreferredHabitatAgent {
public:
    double xi, v_bar;
    PreferredHabitatAgent(double xi_ = 500, double v_bar_ = 1)
      : xi(xi_), v_bar(v_bar_) {}
    double get_action(double pt) {
        return -xi * (pt - v_bar);
    }
};

// --- CircularBuffer and AdaptiveMarketMaker ---
// A circular buffer to store historical data, and a market maker that uses OLS.
class CircularBuffer {
public:
    std::vector<double> buffer;
    int size;
    int index;
    CircularBuffer(int size_) : size(size_), index(0) {
        buffer.resize(size, 0.0);
    }
    void add(double value) {
        buffer[index] = value;
        index = (index + 1) % size;
    }
    std::vector<double> get() {
        std::vector<double> result;
        result.insert(result.end(), buffer.begin() + index, buffer.end());
        result.insert(result.end(), buffer.begin(), buffer.begin() + index);
        return result;
    }
};

class AdaptiveMarketMaker {
public:
    double theta;
    int Tm;
    std::unordered_map<std::string, CircularBuffer*> historical_data;

    AdaptiveMarketMaker(double theta_ = 0.1, int Tm_ = 10000)
      : theta(theta_), Tm(Tm_) {
        std::vector<std::string> vars = {"v", "p", "z", "y"};
        for (const auto& var : vars) {
            historical_data[var] = new CircularBuffer(Tm);
        }
    }
    ~AdaptiveMarketMaker() {
        for (auto& kv : historical_data) {
            delete kv.second;
        }
    }

    // Perform a simple Ordinary Least Squares (OLS) regression.
    // Here we assume a single regressor.
    std::pair<double, double> OLS(const std::string& y_key, const std::string& X_key) {
        std::vector<double> y = historical_data[y_key]->get();
        std::vector<double> X = historical_data[X_key]->get();
        int n = static_cast<int>(y.size());
        double sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
        for (int i = 0; i < n; i++) {
            sumX += X[i];
            sumY += y[i];
            sumXX += X[i] * X[i];
            sumXY += X[i] * y[i];
        }
        double denominator = n * sumXX - sumX * sumX;
        double gamma1 = 0, gamma0 = 0;
        if (std::abs(denominator) > 1e-12) {
            gamma1 = (n * sumXY - sumX * sumY) / denominator;
            gamma0 = (sumY - gamma1 * sumX) / n;
        }
        return {gamma1, gamma0};
    }

    // Determine the market price using the OLS estimates.
    double determine_price(double yt) {
        auto xi_pair = OLS("z", "p");
        auto gamma_pair = OLS("v", "y");
        double xi_1 = xi_pair.first;
        double gamma_1 = gamma_pair.first;
        double gamma_0 = gamma_pair.second;
        double lambda_ = (xi_1 + theta * gamma_1) / (xi_1 * xi_1 + theta);
        double price = gamma_0 + lambda_ * yt;
        return price;
    }

    void update(double vt, double pt, double zt, double yt) {
        historical_data["v"]->add(vt);
        historical_data["p"]->add(pt);
        historical_data["z"]->add(zt);
        historical_data["y"]->add(yt);
    }
};

// --- NoiseAgent Class ---
// Generates a noise (random) action from a normal distribution.
class NoiseAgent {
public:
    double sigma;
    std::mt19937 rng;
    std::normal_distribution<> dist_normal;
    NoiseAgent(double sigma_ = 0.1) : sigma(sigma_), dist_normal(0.0, sigma_) {
        rng.seed(std::random_device{}());
    }
    double get_action() {
        return dist_normal(rng);
    }
};

// --- Global RNG for simplicity ---
std::mt19937 global_rng(std::random_device{}());

// Generate next v using a normal shock.
double get_next_v(double v_bar = 1, double sigma_v = 1) {
    std::normal_distribution<> dist(0.0, sigma_v);
    return v_bar + dist(global_rng);
}

// --- Config Structure ---
struct Config {
    int I;      // number of informed traders
    int Np;     // grid points for price
    int Nv;     // grid points for value
    int Nx;     // grid points for order flow
    double sigma_u;
    double sigma_v;
    double v_bar;
    double xi;
    double theta;
};

// --- Simulation Function ---
// This function runs the simulation for T steps.
void simulate(int T, const Config& config) {
    int I = config.I;
    int Np = config.Np;
    int Nv = config.Nv;
    int Nx = config.Nx;
    double sigma_u = config.sigma_u;

    AdaptiveMarketMaker market_maker(config.theta);
    NoiseAgent noise_agent(sigma_u);
    PreferredHabitatAgent preferred_habitat_agent(config.xi, config.v_bar);
    std::vector<InformedAgent> informed_agents;
    for (int i = 0; i < I; i++) {
        informed_agents.emplace_back(Np, Nv, Nx);
    }

    // Initialize state randomly from the discretized grid.
    std::uniform_int_distribution<> dist_Np(0, Np - 1);
    std::uniform_int_distribution<> dist_Nv(0, Nv - 1);
    std::pair<int,int> state = {dist_Np(global_rng), dist_Nv(global_rng)};

    // History vectors (for example, you could later write these to a file)
    std::vector<double> v_hist(T, 0.0), p_hist(T, 0.0), z_hist(T, 0.0);
    std::vector<std::vector<double>> x_hist(I, std::vector<double>(T, 0.0));
    std::vector<std::vector<double>> y_hist(I, std::vector<double>(T, 0.0));
    std::vector<std::vector<double>> profit_hist(I, std::vector<double>(T, 0.0));

    for (int t = 0; t < T; t++) {
        double _p = informed_agents[0].p_discrete[state.first];
        double _v = informed_agents[0].v_discrete[state.second];
        v_hist[t] = _v;
        p_hist[t] = _p;

        std::vector<double> yt;
        std::vector<int> actions;
        for (int i = 0; i < I; i++) {
            int action = informed_agents[i].get_action(state);
            double xd = informed_agents[i].x_discrete[action];
            yt.push_back(xd);
            actions.push_back(action);
            x_hist[i][t] = xd;
            y_hist[i][t] = xd;
        }

        double yt_sum = 0.0;
        for (double x_val : yt) {
            yt_sum += x_val;
        }
        yt_sum += noise_agent.get_action();

        double zt = preferred_habitat_agent.get_action(_p);
        z_hist[t] = zt;
        market_maker.update(_v, _p, zt, yt_sum);
        double pt = market_maker.determine_price(yt_sum);
        double vt = get_next_v(config.v_bar, config.sigma_v);
        std::pair<int,int> next_state = informed_agents[0].continuous_to_discrete(pt, vt);

        for (int i = 0; i < I; i++) {
            double reward = (_v - pt) * yt[i];
            informed_agents[i].update(state, actions[i], reward, next_state);
            profit_hist[i][t] = reward;
        }
        state = next_state;

        // (Optional) Print progress every so many iterations
        if (t % (T / 10) == 0)
            std::cout << "Simulation progress: " << t * 100 / T << "%\n";
    }
    std::cout << "Simulation completed. Final convergence counter (agent 0): "
              << informed_agents[0].convergence_counter << std::endl;
}

int main() {
    // Set up the configuration parameters
    Config config;
    config.I = 3;       // number of informed traders
    config.Np = 61;
    config.Nv = 20;
    config.Nx = 30;
    config.sigma_u = 0.1;
    config.sigma_v = 1;
    config.v_bar = 1;
    config.xi = 500;
    config.theta = 0.1;

    int T = 500000;  // number of time steps
    simulate(T, config);

    return 0;
}
