// mc_dagprop/monte_carlo/_core.cpp

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <_custom_rng.hpp>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <variant>
#include <vector>

namespace py = pybind11;
using namespace std;

// ── Aliases ───────────────────────────────────────────────────────────────
using EventIndex = int;
using ActivityIndex = int;
using ActivityType = int;
using Second = double;
using Preds = vector<pair<EventIndex, ActivityIndex>>;
using RNG = utl::random::generators::Xoshiro256PP;
using Seed = std::uint64_t;

struct PairHash {
    size_t operator()(const pair<EventIndex, EventIndex> &value) const noexcept {
        return std::hash<EventIndex>{}(value.first) ^ (std::hash<EventIndex>{}(value.second) << 1);
    }
};


static void require_finite_non_negative(double value, const std::string &label) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(label + " must be finite and non-negative");
    }
}

static void require_finite_positive(double value, const std::string &label) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(label + " must be finite and positive");
    }
}

static void require_finite_result(double value, const std::string &label) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(label + " is not finite");
    }
}

static std::vector<double> validate_and_scale_empirical_weights(
    const std::vector<double> &values, const std::vector<double> &weights, const std::string &label) {
    if (values.empty()) {
        throw std::runtime_error(label + ": distribution cannot be empty");
    }
    if (values.size() != weights.size()) {
        throw std::runtime_error(label + ": values and weights must have same length");
    }
    double maximum_weight = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        require_finite_non_negative(values[i], label + ": value at index " + std::to_string(i));
        require_finite_non_negative(weights[i], label + ": weight at index " + std::to_string(i));
        maximum_weight = std::max(maximum_weight, weights[i]);
    }
    if (maximum_weight <= 0.0) {
        throw std::runtime_error(label + ": weights must have positive total mass");
    }

    std::vector<double> scaled_weights;
    scaled_weights.reserve(weights.size());
    for (double weight : weights) {
        scaled_weights.push_back(weight / maximum_weight);
    }
    return scaled_weights;
}

static double uniform_open_unit_interval(RNG &rng) {
    constexpr int MANTISSA_BITS = std::numeric_limits<double>::digits;
    constexpr double UNIT = 1.0 / static_cast<double>(std::uint64_t{1} << MANTISSA_BITS);
    const std::uint64_t mantissa = rng() >> (std::numeric_limits<std::uint64_t>::digits - MANTISSA_BITS);
    return (static_cast<double>(mantissa) + 0.5) * UNIT;
}

static double scaled_product(double first, double second) {
    // frexp/ldexp preserve zero while avoiding an overflowing or underflowing
    // intermediate product for finite non-negative caller inputs.
    int first_exponent;
    int second_exponent;
    const double mantissa = std::frexp(first, &first_exponent) * std::frexp(second, &second_exponent);
    return std::ldexp(mantissa, first_exponent + second_exponent);
}

static double scaled_product(double first, double second, double third) {
    int first_exponent;
    int second_exponent;
    int third_exponent;
    const double mantissa = std::frexp(first, &first_exponent) * std::frexp(second, &second_exponent) *
        std::frexp(third, &third_exponent);
    return std::ldexp(mantissa, first_exponent + second_exponent + third_exponent);
}

static double clip_continuous_sample_to_interior(double sample, double maximum) {
    if (maximum == 0.0) {
        return 0.0;
    }
    if (!std::isfinite(maximum)) {
        return sample;
    }
    // Callers supply non-negative products of validated model parameters.
    return std::min(sample, std::nextafter(maximum, 0.0));
}

static Seed parse_seed(py::handle value) {
    if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) {
        throw py::type_error("seed must be an integer in the range 0..2**64-1 (bool is not accepted)");
    }
    const unsigned long long parsed = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw py::value_error("seed must be an integer in the range 0..2**64-1");
    }
    if (parsed > std::numeric_limits<Seed>::max()) {
        throw py::value_error("seed must be an integer in the range 0..2**64-1");
    }
    return static_cast<Seed>(parsed);
}

static int parse_non_negative_int(py::handle value, const std::string &label) {
    constexpr int MAX_MODEL_INDEX = std::numeric_limits<int>::max();
    if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) {
        throw py::type_error(label + " must be a non-negative integer (bool is not accepted)");
    }
    const long long parsed = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw py::value_error(label + " must not exceed " + std::to_string(MAX_MODEL_INDEX));
    }
    if (parsed < 0) {
        throw py::value_error(label + " must be non-negative");
    }
    if (parsed > MAX_MODEL_INDEX) {
        throw py::value_error(label + " must not exceed " + std::to_string(MAX_MODEL_INDEX));
    }
    return static_cast<int>(parsed);
}

static double parse_real(py::handle value, const std::string &label) {
    if (PyBool_Check(value.ptr())) {
        throw py::type_error(label + " must be a real number (bool is not accepted)");
    }
    try {
        return py::cast<double>(value);
    } catch (const py::cast_error &) {
        throw py::type_error(label + " must be a real number");
    }
}

static std::vector<double> parse_real_sequence(py::iterable values, const std::string &label) {
    std::vector<double> parsed_values;
    for (py::handle value : values) {
        parsed_values.push_back(parse_real(value, label));
    }
    return parsed_values;
}

static std::vector<Seed> parse_seeds(py::iterable values) {
    std::vector<Seed> seeds;
    for (py::handle value : values) {
        seeds.push_back(parse_seed(value));
    }
    return seeds;
}

static double log_standard_normal_cdf(double z_score) {
    constexpr double SQRT_TWO = 1.4142135623730950488;
    constexpr double HALF_LOG_TWO_PI = 0.91893853320467274178;
    constexpr double ASYMPTOTIC_THRESHOLD = -8.0;

    if (z_score > ASYMPTOTIC_THRESHOLD) {
        return std::log(0.5 * std::erfc(-z_score / SQRT_TWO));
    }

    const double magnitude = -z_score;
    const double square = magnitude * magnitude;
    if (!std::isfinite(square)) {
        return -std::numeric_limits<double>::infinity();
    }

    // Mills' ratio: Phi(-x) = phi(x) / x * (1 - 1/x^2 + 3/x^4 - ...).
    const double inverse_square = 1.0 / square;
    double term = 1.0;
    double series = term;
    for (int iteration = 1; iteration <= 64; ++iteration) {
        const double next_term =
            -term * static_cast<double>(2 * iteration - 1) * inverse_square;
        if (std::abs(next_term) >= std::abs(term)) {
            break;
        }
        series += next_term;
        term = next_term;
        if (std::abs(term) <= std::abs(series) * std::numeric_limits<double>::epsilon()) {
            break;
        }
    }
    return -0.5 * square - std::log(magnitude) - HALF_LOG_TWO_PI + std::log(series);
}

static double log_regularized_gamma_p_asymptotic(double shape, double log_value) {
    // The caller uses this approximation only for shape >= 1e7 and
    // value >= shape / 2, so value is positive and the relative difference is
    // greater than -1. An overflowed value naturally maps to a CDF of one.
    const double value = std::exp(log_value);
    const double relative_difference = (value - shape) / shape;
    const double centered_cube_root = std::expm1(std::log1p(relative_difference) / 3.0);
    const double z_score = 3.0 * std::sqrt(shape) * (centered_cube_root + 1.0 / (9.0 * shape));
    return log_standard_normal_cdf(z_score);
}

static double log_regularized_gamma_p_series(double shape, double log_value, double value) {
    constexpr int MAXIMUM_ITERATIONS = 100'000;
    constexpr double RELATIVE_TOLERANCE = 8.0 * std::numeric_limits<double>::epsilon();
    const double log_normalizer = std::lgamma(shape);
    if (!std::isfinite(log_normalizer)) {
        return -std::numeric_limits<double>::infinity();
    }

    double term = 1.0 / shape;
    double sum = term;
    for (int iteration = 1; iteration <= MAXIMUM_ITERATIONS; ++iteration) {
        term *= value / (shape + static_cast<double>(iteration));
        sum += term;
        if (std::abs(term) <= std::abs(sum) * RELATIVE_TOLERANCE) {
            return shape * log_value - value - log_normalizer + std::log(sum);
        }
    }
    throw std::runtime_error("gamma CDF series did not converge");
}

static double log_regularized_gamma_p_continued_fraction(double shape, double log_value, double value) {
    constexpr int MAXIMUM_ITERATIONS = 100'000;
    constexpr double RELATIVE_TOLERANCE = 8.0 * std::numeric_limits<double>::epsilon();
    constexpr double MINIMUM_DENOMINATOR = std::numeric_limits<double>::min() / RELATIVE_TOLERANCE;

    double denominator_term = value + 1.0 - shape;
    if (std::abs(denominator_term) < MINIMUM_DENOMINATOR) {
        denominator_term = MINIMUM_DENOMINATOR;
    }
    double reciprocal_denominator = 1.0 / denominator_term;
    double reciprocal_numerator = 1.0 / MINIMUM_DENOMINATOR;
    double fraction = reciprocal_denominator;

    for (int iteration = 1; iteration <= MAXIMUM_ITERATIONS; ++iteration) {
        const double iteration_value = static_cast<double>(iteration);
        const double numerator = -iteration_value * (iteration_value - shape);
        denominator_term += 2.0;
        reciprocal_denominator = numerator * reciprocal_denominator + denominator_term;
        if (std::abs(reciprocal_denominator) < MINIMUM_DENOMINATOR) {
            reciprocal_denominator = MINIMUM_DENOMINATOR;
        }
        reciprocal_numerator = denominator_term + numerator / reciprocal_numerator;
        if (std::abs(reciprocal_numerator) < MINIMUM_DENOMINATOR) {
            reciprocal_numerator = MINIMUM_DENOMINATOR;
        }
        reciprocal_denominator = 1.0 / reciprocal_denominator;
        const double increment = reciprocal_numerator * reciprocal_denominator;
        fraction *= increment;
        if (std::abs(increment - 1.0) <= RELATIVE_TOLERANCE) {
            const double log_q = shape * log_value - value - std::lgamma(shape) + std::log(fraction);
            if (log_q >= 0.0) {
                return -std::numeric_limits<double>::infinity();
            }
            return std::log1p(-std::exp(log_q));
        }
    }
    throw std::runtime_error("gamma CDF continued fraction did not converge");
}

static double log_regularized_gamma_p(double shape, double log_value) {
    constexpr double ASYMPTOTIC_SHAPE = 10'000'000.0;
    constexpr double LOG_ONE_HALF = -0.6931471805599453;
    constexpr double LOG_MINIMUM_NORMAL = -708.3964185322641;
    if (log_value < LOG_MINIMUM_NORMAL) {
        return shape * log_value - std::lgamma(shape + 1.0);
    }

    const double value = std::exp(log_value);
    if (shape >= ASYMPTOTIC_SHAPE && log_value >= std::log(shape) + LOG_ONE_HALF) {
        return log_regularized_gamma_p_asymptotic(shape, log_value);
    }
    if (value < shape + 1.0) {
        return log_regularized_gamma_p_series(shape, log_value, value);
    }
    return log_regularized_gamma_p_continued_fraction(shape, log_value, value);
}

static double sample_truncated_gamma(RNG &rng, double shape, double scale, double maximum, double duration) {
    constexpr int BISECTION_ITERATIONS = 128;
    constexpr int BRACKETING_ITERATIONS = 2'048;

    const double log_scale = std::log(scale);
    const double scaled_maximum = maximum / scale;
    const double upper_log_value = std::isnormal(scaled_maximum)
        ? std::log(scaled_maximum)
        : std::log(maximum) - log_scale;
    const double maximum_log_probability = log_regularized_gamma_p(shape, upper_log_value);
    const double maximum_seconds = scaled_product(maximum, duration);
    if (maximum_log_probability == -std::numeric_limits<double>::infinity()) {
        // The conditional law is narrower than one representable double at its
        // upper endpoint. Retain the strict interior of the continuous law so
        // floor quantization agrees with the analytic half-open final bin.
        return clip_continuous_sample_to_interior(maximum_seconds, maximum_seconds);
    }
    if (!std::isfinite(maximum_log_probability)) {
        throw std::runtime_error("gamma truncation probability is not numerically representable");
    }
    const double target_log_probability = maximum_log_probability + std::log(uniform_open_unit_interval(rng));

    const double leading_term_estimate =
        (target_log_probability + std::lgamma(shape + 1.0)) / shape;
    double lower_log_value = std::min(upper_log_value - 1.0, leading_term_estimate - 1.0);
    bool bracketed = false;
    for (int iteration = 0; iteration < BRACKETING_ITERATIONS; ++iteration) {
        if (log_regularized_gamma_p(shape, lower_log_value) <= target_log_probability) {
            bracketed = true;
            break;
        }
        lower_log_value -= std::max(1.0, std::abs(lower_log_value) * 0.5);
    }
    if (!bracketed) {
        throw std::runtime_error("could not bracket truncated gamma quantile");
    }

    double upper = upper_log_value;
    double lower = lower_log_value;
    for (int iteration = 0; iteration < BISECTION_ITERATIONS; ++iteration) {
        const double midpoint = lower + (upper - lower) * 0.5;
        if (log_regularized_gamma_p(shape, midpoint) < target_log_probability) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }

    const double sample = std::exp(lower + log_scale + std::log(duration));
    return clip_continuous_sample_to_interior(sample, maximum_seconds);
}

// ── Core Data Types ───────────────────────────────────────────────────────
struct EventTimestamp {
    double earliest, latest, actual;
};

struct Event {
    std::string event_id;
    EventTimestamp ts;
};

struct Activity {
    ActivityIndex idx;
    Second duration;
    ActivityType activity_type;
};

// ── Simulation Context ───────────────────────────────────────────────────
struct DagContext {
    vector<Event> events;
    unordered_map<pair<EventIndex, EventIndex>, Activity, PairHash> activity_map;
    vector<pair<EventIndex, Preds>> precedence_list;
    DagContext(vector<Event> ev, unordered_map<pair<EventIndex, EventIndex>, Activity, PairHash> am,
               vector<pair<EventIndex, Preds>> pl)
        : events(std::move(ev)), activity_map(std::move(am)), precedence_list(std::move(pl)) {}
};

// ── Simulation Result ────────────────────────────────────────────────────
struct SimResult {
    vector<double> realized;
    vector<double> durations;
    vector<EventIndex> cause_event;
};

// ── Delay Distributions ──────────────────────────────────────────────────
struct ConstantDist {
    double factor;
    ConstantDist(const double f = 0.0) : factor(f) { require_finite_non_negative(f, "constant delay factor"); }
    double sample(RNG &, const double d) const { return d * factor; }
};

struct ExponentialDist {
    double scale, max_scale;
    ExponentialDist(const double sc = 1.0, const double mx = 1.0) : scale(sc), max_scale(mx) {
        require_finite_positive(sc, "exponential scale");
        require_finite_positive(mx, "exponential max_scale");
    }
    double sample(RNG &rng, const double d) const {
        if (d == 0.0) {
            return 0.0;
        }
        const double truncation_ratio = max_scale / scale;
        const double maximum_seconds = scaled_product(max_scale, d);
        constexpr double UNIFORM_APPROXIMATION_THRESHOLD = 1.4901161193847656e-8;
        if (truncation_ratio <= UNIFORM_APPROXIMATION_THRESHOLD) {
            const double sample = scaled_product(uniform_open_unit_interval(rng), max_scale, d);
            return clip_continuous_sample_to_interior(sample, maximum_seconds);
        }
        const double truncation_probability = -std::expm1(-truncation_ratio);
        const double unit_scale_sample = -std::log1p(-uniform_open_unit_interval(rng) * truncation_probability);
        const double sample = scaled_product(unit_scale_sample, scale, d);
        return clip_continuous_sample_to_interior(sample, maximum_seconds);
    }
};

struct GammaDist {
    double shape, scale, max_scale;
    GammaDist(const double k = 1.0, const double s = 1.0, const double m = 10.0)
        : shape(k), scale(s), max_scale(m) {
        require_finite_positive(k, "gamma shape");
        require_finite_positive(s, "gamma scale");
        require_finite_positive(m, "gamma max_scale");
    }
    double sample(RNG &rng, const double d) const {
        if (d == 0.0) {
            return 0.0;
        }
        constexpr int FAST_PATH_ATTEMPTS = 8;
        const double maximum_unit_scale = max_scale / scale;
        const double maximum_seconds = scaled_product(max_scale, d);
        std::gamma_distribution<double> distribution(shape, 1.0);
        for (int attempt = 0; attempt < FAST_PATH_ATTEMPTS; ++attempt) {
            const double sample = distribution(rng);
            if (std::isfinite(sample) && sample <= maximum_unit_scale) {
                const double sample_seconds = scaled_product(sample, scale, d);
                return clip_continuous_sample_to_interior(sample_seconds, maximum_seconds);
            }
        }
        return sample_truncated_gamma(rng, shape, scale, max_scale, d);
    }
};

// ── Empirical “table” distributions ─────────────────────────────────────

// 1) Absolute: user‐supplied values are taken literally
struct EmpiricalAbsoluteDist {
    std::vector<double> values;
    std::discrete_distribution<size_t> dist;

    EmpiricalAbsoluteDist(std::vector<double> vals, std::vector<double> weights)
        : values(std::move(vals))
          ,
          dist() {
        const std::vector<double> scaled_weights =
            validate_and_scale_empirical_weights(values, weights, "EmpiricalAbsoluteDist");
        dist = std::discrete_distribution<size_t>(scaled_weights.begin(), scaled_weights.end());
    }

    // Sampling draws from the distribution and mutates its state
    double sample(RNG &rng, double /*duration*/) { return values[dist(rng)]; }
};

// 2) Relative: user‐supplied factors in [0..∞), multiplied by the base duration
struct EmpiricalRelativeDist {
    std::vector<double> factors;
    std::discrete_distribution<size_t> dist;

    EmpiricalRelativeDist(std::vector<double> facs, std::vector<double> weights) : factors(std::move(facs)), dist() {
        const std::vector<double> scaled_weights =
            validate_and_scale_empirical_weights(factors, weights, "EmpiricalRelativeDist");
        dist = std::discrete_distribution<size_t>(scaled_weights.begin(), scaled_weights.end());
    }

    // Sampling draws from the distribution and mutates its state
    double sample(RNG &rng, double duration) { return factors[dist(rng)] * duration; }
}; 

using DistVar = std::variant<ConstantDist, ExponentialDist, GammaDist, EmpiricalAbsoluteDist, EmpiricalRelativeDist>;

// ── Delay Generator ──────────────────────────────────────────────────────
class GenericDelayGenerator {
   public:
    unordered_map<ActivityType, DistVar> dist_map_;

    GenericDelayGenerator() = default;
    void validate_activity_type(ActivityType t) const {
        if (t < 0) {
            throw std::runtime_error("activity type " + std::to_string(t) + " is reserved and cannot be registered");
        }
    }
    void ensure_unregistered(ActivityType t) const {
        if (dist_map_.count(t)) {
            throw std::runtime_error("delay family already registered for activity type " + std::to_string(t));
        }
    }
    void add_constant(ActivityType t, double f) { validate_activity_type(t); ensure_unregistered(t); dist_map_[t] = ConstantDist{f}; }
    void add_exponential(ActivityType t, double scale, double mx) { validate_activity_type(t); ensure_unregistered(t); dist_map_[t] = ExponentialDist{scale, mx}; }
    void add_gamma(ActivityType t, double k, double s, double m = 10.0) {
        validate_activity_type(t);
        ensure_unregistered(t);
        dist_map_[t] = GammaDist{k, s, m};
    }
};

// ── Simulator ────────────────────────────────────────────────────────────
class Simulator {
    DagContext context_;

    // Delay distributions (flattened)
    std::vector<DistVar> delay_distributions_;
    std::unordered_map<ActivityType, int> activity_type_to_dist_index_;

    // Activities: one per link index
    std::vector<Activity> activities_;
    std::vector<int> activity_to_dist_index_;  // -1 = no delay

    // Precedence (CSR format)
    std::vector<EventIndex> flat_predecessor_sources_;  // all predecessor node indices
    std::vector<ActivityIndex> flat_predecessor_edges_;    // corresponding edge indices
    std::vector<size_t> predecessor_offsets_;         // prefix offsets per event
    std::vector<EventIndex> event_evaluation_order_;    // order in which to process events

    // Sampler function: signature(sample_rng, distribution_variant, base_duration)
    using SamplerFunc = double(*)(RNG&, DistVar&, double);
    std::vector<SamplerFunc> sampler_functions_;

    void validate_context() const {
        const int event_count = int(context_.events.size());
        if (event_count == 0) {
            throw std::runtime_error("DAG context must contain at least one event");
        }
        std::unordered_set<std::string> seen_event_ids;
        for (int i = 0; i < event_count; ++i) {
            if (!seen_event_ids.insert(context_.events[i].event_id).second) {
                throw std::runtime_error("duplicate event id " + context_.events[i].event_id);
            }
            const auto &ts = context_.events[i].ts;
            if (!std::isfinite(ts.earliest) || !std::isfinite(ts.latest) || !std::isfinite(ts.actual)) {
                throw std::runtime_error("event " + std::to_string(i) + " times must be finite");
            }
            if (ts.earliest > ts.actual || ts.actual > ts.latest) {
                throw std::runtime_error(
                    "event " + std::to_string(i) + " times must satisfy earliest <= actual <= latest");
            }
        }
        std::unordered_set<ActivityIndex> seen_activity_indices;
        for (const auto &kv : context_.activity_map) {
            const EventIndex src = kv.first.first;
            const EventIndex dst = kv.first.second;
            const Activity &activity = kv.second;
            if (src < 0 || src >= event_count || dst < 0 || dst >= event_count) {
                throw std::runtime_error("activity " + std::to_string(activity.idx) + " references invalid event index");
            }
            if (activity.idx < 0) {
                throw std::runtime_error("activity index " + std::to_string(activity.idx) + " must be non-negative");
            }
            if (activity.activity_type < 0) {
                throw std::runtime_error("activity " + std::to_string(activity.idx) + " has reserved negative activity type");
            }
            require_finite_non_negative(activity.duration, "activity " + std::to_string(activity.idx) + " minimal_duration");
            if (!seen_activity_indices.insert(activity.idx).second) {
                throw std::runtime_error("duplicate activity index " + std::to_string(activity.idx));
            }
        }
        for (int expected = 0; expected < int(seen_activity_indices.size()); ++expected) {
            if (!seen_activity_indices.count(expected)) {
                throw std::runtime_error("activity indices must be contiguous from 0 to n-1");
            }
        }
        std::unordered_set<EventIndex> seen_targets;
        std::unordered_set<ActivityIndex> referenced_activity_indices;
        for (const auto &entry : context_.precedence_list) {
            EventIndex target = entry.first;
            if (target < 0 || target >= event_count) {
                throw std::runtime_error("target index " + std::to_string(target) + " out of range");
            }
            if (!seen_targets.insert(target).second) {
                throw std::runtime_error("duplicate precedence entry for target " + std::to_string(target));
            }
            std::unordered_set<EventIndex> seen_predecessor_sources;
            for (const auto &pred : entry.second) {
                EventIndex source = pred.first;
                ActivityIndex activity_index = pred.second;
                if (source < 0 || source >= event_count) {
                    throw std::runtime_error("predecessor index " + std::to_string(source) + " out of range");
                }
                if (!seen_predecessor_sources.insert(source).second) {
                    throw std::runtime_error(
                        "duplicate predecessor " + std::to_string(source) + " for target " + std::to_string(target));
                }
                auto edge = context_.activity_map.find({source, target});
                if (edge == context_.activity_map.end()) {
                    throw std::runtime_error("missing activity for predecessor edge");
                }
                if (edge->second.idx != activity_index) {
                    throw std::runtime_error("precedence activity id " + std::to_string(activity_index) + " does not match edge");
                }
                referenced_activity_indices.insert(activity_index);
            }
        }
        if (referenced_activity_indices.size() != context_.activity_map.size()) {
            throw std::runtime_error("every activity must be referenced by exactly one precedence edge");
        }
    }

public:
    Simulator(DagContext context, GenericDelayGenerator generator)
        : context_(std::move(context)) {
        validate_context();
        // 0) Validate reserved activity_type
        if (generator.dist_map_.count(-1)) {
            throw std::runtime_error("Activity type -1 is reserved for no delay");
        }
        // 1) Flatten distributions and build type->index map
        delay_distributions_.reserve(generator.dist_map_.size());
        int dist_counter = 0;
        for (auto &entry : generator.dist_map_) {
            activity_type_to_dist_index_[entry.first] = dist_counter;
            delay_distributions_.push_back(entry.second);
            ++dist_counter;
        }

        // 2) Allocate activities and map each link to a distribution index
        int max_link_index = -1;
        for (auto &kv : context_.activity_map) {
            max_link_index = std::max(max_link_index, kv.second.idx);
        }
        int link_count = max_link_index + 1;
        activities_.assign(link_count, Activity{ActivityIndex(-1), Second(0.0), ActivityType(-1)});
        activity_to_dist_index_.assign(link_count, -1);

        for (auto &kv : context_.activity_map) {
            const Activity &edge = kv.second;
            ActivityIndex link_idx = edge.idx;
            activities_[link_idx] = edge;
            auto it = activity_type_to_dist_index_.find(edge.activity_type);
            if (it != activity_type_to_dist_index_.end()) {
                activity_to_dist_index_[link_idx] = it->second;
            }
        }

        // 3) Build CSR for precedences and compute topological order
        int event_count = int(context_.events.size());

        // build adjacency list and indegree counters
        std::vector<std::vector<EventIndex>> adjacency(event_count);
        std::vector<Preds> preds_by_target(event_count);
        std::vector<int> indegree(event_count, 0);
        for (auto &entry : context_.precedence_list) {
            EventIndex tgt = entry.first;
            preds_by_target[tgt] = entry.second;
            indegree[tgt] = int(entry.second.size());
            for (auto &pr : entry.second) {
                adjacency[pr.first].push_back(tgt);
            }
        }

        // Kahn's algorithm for topological sorting
        event_evaluation_order_.clear();
        event_evaluation_order_.reserve(event_count);
        std::deque<EventIndex> q;
        for (int i = 0; i < event_count; ++i) {
            if (indegree[i] == 0) q.push_back(i);
        }
        while (!q.empty()) {
            EventIndex n = q.front();
            q.pop_front();
            event_evaluation_order_.push_back(n);
            for (EventIndex dst : adjacency[n]) {
                if (--indegree[dst] == 0) q.push_back(dst);
            }
        }
        if ((int)event_evaluation_order_.size() != event_count) {
            throw std::runtime_error("Invalid DAG: cycle detected in precedence list");
        }

        // build CSR arrays using sorted order
        predecessor_offsets_.assign(event_count + 1, 0);
        for (int i = 0; i < event_count; ++i) {
            predecessor_offsets_[i + 1] = preds_by_target[i].size();
        }
        for (int i = 1; i <= event_count; ++i) {
            predecessor_offsets_[i] += predecessor_offsets_[i - 1];
        }
        int total_predecessors = predecessor_offsets_[event_count];
        flat_predecessor_sources_.resize(total_predecessors);
        flat_predecessor_edges_.resize(total_predecessors);

        std::vector<size_t> write_positions = predecessor_offsets_;
        for (EventIndex event_id : event_evaluation_order_) {
            for (auto &pr : preds_by_target[event_id]) {
                size_t idx = write_positions[event_id]++;
                flat_predecessor_sources_[idx] = pr.first;
                flat_predecessor_edges_[idx] = pr.second;
            }
        }

        // 4) Prepare sampler function pointers
        sampler_functions_.resize(link_count, nullptr);
        for (int link = 0; link < link_count; ++link) {
            int dist_idx = activity_to_dist_index_[link];
            if (dist_idx < 0) continue;
            sampler_functions_[link] = [](RNG &rng, DistVar &var, double base_dur) {
                return std::visit([&](auto &dist) { return dist.sample(rng, base_dur); }, var);
            };
        }

    }

    inline int node_count() const noexcept { return int(context_.events.size()); }
    inline int activity_count() const noexcept { return int(activities_.size()); }

    SimResult run(Seed seed) const {
        RNG rng(seed);
        std::vector<DistVar> delay_distributions = delay_distributions_;
        const int event_count = node_count();
        const int activity_count = this->activity_count();
        SimResult result{
            std::vector<double>(event_count),
            std::vector<double>(activity_count),
            std::vector<EventIndex>(event_count, -1),
        };

        for (int event_index = 0; event_index < event_count; ++event_index) {
            result.realized[event_index] = context_.events[event_index].ts.earliest;
        }

        for (auto &dist : delay_distributions) {
            std::visit([](auto &typed_dist) {
                using DistType = std::decay_t<decltype(typed_dist)>;
                if constexpr (std::is_same_v<DistType, EmpiricalAbsoluteDist> ||
                              std::is_same_v<DistType, EmpiricalRelativeDist>) {
                    typed_dist.dist.reset();
                }
            }, dist);
        }

        for (int link = 0; link < activity_count; ++link) {
            int dist_idx = activity_to_dist_index_[link];
            double base_dur = activities_[link].duration;
            if (dist_idx < 0) {
                result.durations[link] = base_dur;
                continue;
            }
            const double extra = sampler_functions_[link](rng, delay_distributions[dist_idx], base_dur);
            require_finite_result(extra, "sampled extra delay for activity " + std::to_string(link));
            if (extra < 0.0) {
                throw std::runtime_error("sampled extra delay for activity " + std::to_string(link) + " is negative");
            }
            result.durations[link] = base_dur + extra;
            require_finite_result(result.durations[link], "realized duration for activity " + std::to_string(link));
        }

        for (EventIndex event_id : event_evaluation_order_) {
            double latest = result.realized[event_id];
            EventIndex cause = -1;
            for (size_t idx = predecessor_offsets_[event_id]; idx < predecessor_offsets_[event_id + 1]; ++idx) {
                EventIndex src = flat_predecessor_sources_[idx];
                ActivityIndex edge = flat_predecessor_edges_[idx];
                const double t = result.realized[src] + result.durations[edge];
                require_finite_result(
                    t,
                    "propagated time from event " + std::to_string(src) + " to event " + std::to_string(event_id));
                if (t >= latest) {
                    latest = t;
                    cause = src;
                }
            }
            result.realized[event_id] = latest;
            result.cause_event[event_id] = cause;
        }

        return result;
    }

    std::vector<SimResult> run_many(const std::vector<Seed> &seeds) const {
        std::vector<SimResult> results;
        results.reserve(seeds.size());
        for (Seed seed : seeds) results.emplace_back(run(seed));
        return results;
    }
};


// ── Python Bindings ─────────────────────────────────────────────────────
PYBIND11_MODULE(_core, m) {
    m.doc() = "Core Monte-Carlo DAG-propagation simulator";

    // EventTimestamp
    py::class_<EventTimestamp> ts_cls(m, "EventTimestamp");
    ts_cls
        .def(py::init([](py::handle earliest, py::handle latest, py::handle actual) {
                return EventTimestamp{
                    parse_real(earliest, "earliest"),
                    parse_real(latest, "latest"),
                    parse_real(actual, "actual"),
                };
             }), py::arg("earliest"), py::arg("latest"), py::arg("actual"),
             "Store timestamp metadata; propagation-context validation enforces finite ordered values.")
        .def_readonly("earliest", &EventTimestamp::earliest, "Initial realized time and lower bound")
        .def_readonly("latest", &EventTimestamp::latest, "Upper-bound metadata; Monte Carlo does not clip to it")
        .def_readonly("actual", &EventTimestamp::actual, "External/reference timestamp; not a propagation input")
        .def(
            "__repr__",
            [](const EventTimestamp &ts) {
                return py::str("EventTimestamp(earliest={}, latest={}, actual={})")
                    .format(ts.earliest, ts.latest, ts.actual);
            },
            "Return ``repr(self)`` style string.");

    // Event
    py::class_<Event> event_cls(m, "Event");
    event_cls
        .def(py::init<std::string, EventTimestamp>(), py::arg("event_id"), py::arg("timestamp"),
             "An event node with its ID and timestamp")
        .def_readonly("event_id", &Event::event_id, "Node identifier")
        .def_readonly("timestamp", &Event::ts, "Event timing info")
        .def(
            "__repr__",
            [](const Event &ev) {
                py::object id_r = py::repr(py::cast(ev.event_id));
                py::object ts_r = py::repr(py::cast(ev.ts));
                return py::str("Event(event_id={}, timestamp={})").format(id_r, ts_r);
            },
            "Return ``repr(self)`` style string.");

    // Activity
    py::class_<Activity> activity_cls(m, "Activity");
    activity_cls
        .def(
            py::init([](py::handle index, py::handle minimal_duration, py::handle activity_type) {
                return Activity{
                    parse_non_negative_int(index, "activity index"),
                    parse_real(minimal_duration, "minimal_duration"),
                    parse_non_negative_int(activity_type, "activity_type"),
                };
            }),
             py::arg("idx"),
             py::arg("minimal_duration"),
             py::arg("activity_type"),
             "An activity (edge) with index, base duration and type")
        .def_readonly("idx", &Activity::idx, "Index of the activity")
        .def_readonly("minimal_duration", &Activity::duration, "Base duration")
        .def_readonly("activity_type", &Activity::activity_type, "Type ID for delay dist.")
        .def(
            "__repr__",
            [](const Activity &act) {
                return py::str(
                           "Activity(idx={}, minimal_duration={}, activity_type={})")
                    .format(act.idx, act.duration, act.activity_type);
            },
            "Return ``repr(self)`` style string.");

    // DagContext
    py::class_<DagContext> ctx_cls(m, "DagContext");
    ctx_cls
        .def(py::init([](py::sequence events_input, py::object activities_input, py::sequence precedence_input) {
                std::vector<Event> events;
                events.reserve(events_input.size());
                for (py::handle event : events_input) {
                    events.push_back(py::cast<Event>(event));
                }

                std::unordered_map<pair<EventIndex, EventIndex>, Activity, PairHash> activities;
                py::iterable activity_items = activities_input.attr("items")();
                for (py::handle item : activity_items) {
                    const py::tuple entry = py::cast<py::tuple>(item);
                    const py::tuple endpoints = py::cast<py::tuple>(entry[0]);
                    if (endpoints.size() != 2) {
                        throw py::value_error("activity keys must contain exactly two event indices");
                    }
                    const EventIndex source = parse_non_negative_int(endpoints[0], "activity source index");
                    const EventIndex target = parse_non_negative_int(endpoints[1], "activity target index");
                    activities.emplace(std::make_pair(source, target), py::cast<Activity>(entry[1]));
                }

                std::vector<pair<EventIndex, Preds>> precedence_list;
                precedence_list.reserve(precedence_input.size());
                for (py::handle item : precedence_input) {
                    const py::tuple entry = py::cast<py::tuple>(item);
                    if (entry.size() != 2) {
                        throw py::value_error("precedence entries must contain a target and predecessor sequence");
                    }
                    const EventIndex target = parse_non_negative_int(entry[0], "precedence target index");
                    const py::sequence predecessors_input = py::cast<py::sequence>(entry[1]);
                    Preds predecessors;
                    predecessors.reserve(predecessors_input.size());
                    for (py::handle predecessor : predecessors_input) {
                        const py::tuple predecessor_entry = py::cast<py::tuple>(predecessor);
                        if (predecessor_entry.size() != 2) {
                            throw py::value_error(
                                "predecessor entries must contain a source and activity index");
                        }
                        predecessors.emplace_back(
                            parse_non_negative_int(predecessor_entry[0], "predecessor source index"),
                            parse_non_negative_int(predecessor_entry[1], "predecessor activity index"));
                    }
                    precedence_list.emplace_back(target, std::move(predecessors));
                }
                return DagContext{std::move(events), std::move(activities), std::move(precedence_list)};
             }),
             py::arg("events"), py::arg("activities"), py::arg("precedence_list"),
             "Wraps a DAG: events, activity_map, precedence_list")
        .def_readonly("events", &DagContext::events)
        .def_readonly("activities", &DagContext::activity_map)
        .def_readonly("precedence_list", &DagContext::precedence_list)
        .def(
            "__repr__",
            [](const DagContext &ctx) {
                py::object events_r = py::repr(py::cast(ctx.events));
                py::object act_r = py::repr(py::cast(ctx.activity_map));
                py::object preds_r = py::repr(py::cast(ctx.precedence_list));
                return py::str("DagContext(events={}, activities={}, precedence_list={})")
                    .format(events_r, act_r, preds_r);
            },
            "Return ``repr(self)`` style string.");

    // Turn core structs into frozen dataclasses
    py::object dataclass_fn = py::module_::import("dataclasses").attr("dataclass");
    py::dict dc_opts;
    dc_opts["frozen"] = true;
    dc_opts["slots"] = true;
    dc_opts["init"] = false;
    py::object dataclass = dataclass_fn(**dc_opts);

    py::module types_mod = py::module_::import("mc_dagprop.types");
    py::object Second = types_mod.attr("Second");
    py::object py_EventId = types_mod.attr("EventId");
    py::object py_ActivityIndex = types_mod.attr("ActivityIndex");
    py::object py_ActivityType = types_mod.attr("ActivityType");

    py::dict ts_ann;
    ts_ann["earliest"] = Second;
    ts_ann["latest"] = Second;
    ts_ann["actual"] = Second;
    ts_cls.attr("__annotations__") = ts_ann;
    dataclass(ts_cls);

    py::dict ev_ann;
    ev_ann["event_id"] = py_EventId;
    ev_ann["timestamp"] = ts_cls;
    event_cls.attr("__annotations__") = ev_ann;
    dataclass(event_cls);

    py::dict act_ann;
    act_ann["idx"] = py_ActivityIndex;
    act_ann["minimal_duration"] = Second;
    act_ann["activity_type"] = py_ActivityType;
    activity_cls.attr("__annotations__") = act_ann;
    dataclass(activity_cls);

    py::object typing = py::module_::import("typing");
    py::object Sequence = typing.attr("Sequence");
    py::object Mapping = typing.attr("Mapping");
    py::dict ctx_ann;
    ctx_ann["events"] = Sequence;
    ctx_ann["activities"] = Mapping;
    ctx_ann["precedence_list"] = Sequence;
    ctx_cls.attr("__annotations__") = ctx_ann;
    dataclass(ctx_cls);

    // SimResult // SimResult → return NumPy arrays instead of lists
    py::class_<SimResult>(m, "SimResult", py::buffer_protocol())
        .def_buffer([](SimResult &r) -> py::buffer_info {
            return py::buffer_info(r.realized.data(),                        // Pointer to buffer
                                   sizeof(double),                           // Size of one scalar
                                   py::format_descriptor<double>::format(),  // Python struct-style format descriptor
                                   1,                                        // Number of dimensions
                                   {r.realized.size()},                      // Buffer dimensions
                                   {sizeof(double)}                          // Strides (in bytes) for each index
            );
        })
        .def_property_readonly(
            "realized",
            [](const SimResult &r) {
                return py::array(r.realized.size(),  // shape
                                 r.realized.data(),  // pointer to data
                                 py::cast(r)         // capsule to ensure SimResult stays alive
                );
            },
            "Final event times as a NumPy array")
        .def_property_readonly(
            "durations", [](SimResult &r) { return py::array(r.durations.size(), r.durations.data(), py::cast(r)); },
            "Per-link durations (incl. extra) as a NumPy array")
        .def_property_readonly(
            "cause_event",
            [](SimResult &r) { return py::array(r.cause_event.size(), r.cause_event.data(), py::cast(r)); },
            "Index of predecessor causing each event as a NumPy array");

    // GenericDelayGenerator
    py::class_<GenericDelayGenerator>(m, "GenericDelayGenerator")
        .def(py::init<>(), "Create a new delay‐generator")
        .def("add_constant", [](GenericDelayGenerator &generator, py::handle activity_type, py::handle factor) {
                generator.add_constant(
                    parse_non_negative_int(activity_type, "activity_type"),
                    parse_real(factor, "constant delay factor"));
             }, py::arg("activity_type"), py::arg("factor"),
             "Constant extra delay: factor * minimal_duration")
        .def("add_exponential", [](GenericDelayGenerator &generator, py::handle activity_type, py::handle scale,
                                   py::handle max_scale) {
                generator.add_exponential(
                    parse_non_negative_int(activity_type, "activity_type"),
                    parse_real(scale, "exponential scale"),
                    parse_real(max_scale, "exponential max_scale"));
             }, py::arg("activity_type"), py::arg("scale"),
             py::arg("max_scale"),
             "Dimensionless exponential factor with mean scale, conditioned on factor <= max_scale")
        .def("add_gamma", [](GenericDelayGenerator &generator, py::handle activity_type, py::handle shape,
                             py::handle scale, py::handle max_scale) {
                generator.add_gamma(
                    parse_non_negative_int(activity_type, "activity_type"),
                    parse_real(shape, "gamma shape"),
                    parse_real(scale, "gamma scale"),
                    parse_real(max_scale, "gamma max_scale"));
             }, py::arg("activity_type"), py::arg("shape"),
             py::arg("scale"), py::arg("max_scale") = 10.0,
             "Dimensionless Gamma(shape, scale) factor, conditioned on factor <= max_scale")
        .def(
            "add_empirical_absolute",
            [](GenericDelayGenerator &g, py::handle activity_type, py::iterable values, py::iterable weights) {
                const int parsed_activity_type = parse_non_negative_int(activity_type, "activity_type");
                g.ensure_unregistered(parsed_activity_type);
                g.dist_map_[parsed_activity_type] = EmpiricalAbsoluteDist{
                    parse_real_sequence(std::move(values), "empirical value"),
                    parse_real_sequence(std::move(weights), "empirical weight"),
                };
            },
            py::arg("activity_type"), py::arg("values"), py::arg("weights"),
            "Draw an absolute extra delay in seconds from the weighted values.")
        .def(
            "add_empirical_relative",
            [](GenericDelayGenerator &g, py::handle activity_type, py::iterable factors, py::iterable weights) {
                const int parsed_activity_type = parse_non_negative_int(activity_type, "activity_type");
                g.ensure_unregistered(parsed_activity_type);
                g.dist_map_[parsed_activity_type] = EmpiricalRelativeDist{
                    parse_real_sequence(std::move(factors), "empirical factor"),
                    parse_real_sequence(std::move(weights), "empirical weight"),
                };
            },
            py::arg("activity_type"), py::arg("factors"), py::arg("weights"),
            "Draw a dimensionless factor, then multiply it by minimal_duration.");

    // Simulator
    py::class_<Simulator>(m, "MonteCarloPropagator")
        .def(py::init<DagContext, GenericDelayGenerator>(), py::arg("context"), py::arg("generator"),
             "Construct a reentrant, thread-safe propagator with immutable model state")
        .def_static(
            "from_context",
            [](py::object context, py::object registry) {
                return py::module_::import("mc_dagprop.frontend")
                    .attr("monte_carlo_from_context")(std::move(context), std::move(registry));
            },
            py::arg("context"), py::arg("registry"),
            "Construct from the shared PropagationContext and DelayFamilyRegistry frontend.")
        .def("node_count", &Simulator::node_count, "Number of events")
        .def("activity_count", &Simulator::activity_count, "Number of links")
        .def(
            "run",
            [](const Simulator &simulator, py::handle seed) {
                const Seed parsed_seed = parse_seed(seed);
                SimResult result;
                {
                    py::gil_scoped_release release;
                    result = simulator.run(parsed_seed);
                }
                return result;
            },
            py::arg("seed"),
            "Run one simulation reproducibly within a fixed package build/platform; concurrent calls are safe.")
        .def(
            "run_many",
            [](const Simulator &simulator, py::iterable seeds) {
                const std::vector<Seed> parsed_seeds = parse_seeds(std::move(seeds));
                std::vector<SimResult> results;
                {
                    py::gil_scoped_release release;
                    results = simulator.run_many(parsed_seeds);
                }
                return results;
            },
            py::arg("seeds"),
            "Run seeds in order, exactly as independent run(seed) calls; concurrent calls are safe.");
}
