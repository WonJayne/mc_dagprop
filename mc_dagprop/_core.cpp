#include <algorithm>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>

namespace py = pybind11;

// ------------------------- Custom Hash for pair -------------------------
namespace std {
    template <typename T1, typename T2>
    struct hash<std::pair<T1, T2>> {
        size_t operator()(const std::pair<T1, T2>& p) const {
            return std::hash<T1>{}(p.first) ^ (std::hash<T2>{}(p.second) << 1);
        }
    };
}

// ------------------------- Data Types -------------------------
struct EventPointInTime {
    double earliest, latest, actual;
};

struct SimulationTreeLink {
    double minimal_duration;
    int link_type;
};

using NodeId = std::string;
using LinkIndex = int;
using NodeIndex = int;
using Precedence = std::vector<std::pair<NodeIndex, LinkIndex>>;

struct SimContext {
    std::vector<std::pair<NodeId, EventPointInTime>> events;
    std::unordered_map<std::pair<NodeIndex, NodeIndex>, std::pair<LinkIndex, SimulationTreeLink>> link_map;
    std::vector<std::pair<NodeIndex, Precedence>> precedence_list;
    double max_delay;

    SimContext(
        std::vector<std::pair<NodeId, EventPointInTime>> ev,
        std::unordered_map<std::pair<NodeIndex, NodeIndex>, std::pair<LinkIndex, SimulationTreeLink>> lm,
        std::vector<std::pair<NodeIndex, Precedence>> pl,
        double max_delay_
    ) : events(std::move(ev)), link_map(std::move(lm)), precedence_list(std::move(pl)), max_delay(max_delay_) {}
};

// ------------------------- SimResult -------------------------
struct SimResult {
    std::vector<double> realized;
    std::vector<double> delays;
    std::vector<int>    cause_event;
};

// ------------------------- Delay Distributions -------------------------
struct ConstantDist {
    double factor;
    ConstantDist(double f): factor(f) {}
    double sample(std::mt19937& /*rng*/, double minimal_duration) const {
        return minimal_duration * factor;
    }
};

struct ExponentialDist {
    double lambda;
    double max_scale;
    std::exponential_distribution<double> dist;
    ExponentialDist(double lam, double max_s)
      : lambda(lam), max_scale(max_s), dist(1.0/lam) {}
    double sample(std::mt19937& rng, double minimal_duration) const {
        double x;
        do {
            x = dist(rng);
        } while (x > max_scale);
        return x * minimal_duration;
    }
};

// ------------------------- Generic Delay Generator -------------------------
class GenericDelayGenerator {
    std::mt19937 rng_;
    using DistVariant = std::variant<ConstantDist, ExponentialDist>;
    std::unordered_map<int, DistVariant> dist_map_;

public:
    GenericDelayGenerator(): rng_(std::random_device{}()) {}

    void inline set_seed(int seed) {
        rng_.seed(seed);
    }

    void add_constant(int link_type, double factor) {
        dist_map_.insert_or_assign(link_type, ConstantDist{factor});
    }

    void add_exponential(int link_type, double lambda, double max_scale) {
        dist_map_.insert_or_assign(link_type, ExponentialDist{lambda, max_scale});
    }

    double get_delay(const SimulationTreeLink& link) {
        auto it = dist_map_.find(link.link_type);
        if (it == dist_map_.end()) return 0.0;
        return std::visit([&](auto& d) {
            return d.sample(rng_, link.minimal_duration);
        }, it->second);
    }
};

// ------------------------- Simulator -------------------------
class Simulator {
    SimContext            context_;
    GenericDelayGenerator generator_;
    std::vector<SimulationTreeLink> links_;
    std::vector<bool>               is_affected_;
    std::vector<double>             durations_;

public:
    Simulator(SimContext ctx, GenericDelayGenerator gen)
      : context_(std::move(ctx)), generator_(std::move(gen)) {
        int nlinks = (int)context_.link_map.size();
        links_.resize(nlinks);
        is_affected_.resize(nlinks, false);
        durations_.resize(nlinks);
        for (auto& kv : context_.link_map) {
            int idx = kv.second.first;
            auto link = kv.second.second;
            links_[idx]      = link;
            is_affected_[idx] = (link.link_type == 1 && link.minimal_duration >= 0.01);
            durations_[idx]   = link.minimal_duration;
        }
    }

    SimResult run(int seed) {
        generator_.set_seed(seed);
        int ne = (int)context_.events.size();
        int nl = (int)links_.size();

        std::vector<double> lower(ne), scheduled(ne);
        for (int i = 0; i < ne; ++i) {
            lower[i]     = context_.events[i].second.earliest;
            scheduled[i] = context_.events[i].second.actual;
        }

        std::vector<double> compounded(nl);
        for (int i = 0; i < nl; ++i) {
            compounded[i] = durations_[i] + (is_affected_[i] ? generator_.get_delay(links_[i]) : 0.0);
        }

        std::vector<double> realized = lower;
        std::vector<int>    cause_event(ne);
        std::iota(cause_event.begin(), cause_event.end(), 0);

        for (auto& p : context_.precedence_list) {
            int n_idx = p.first;
            auto& preds = p.second;
            if (preds.size() == 1) {
                int pi = preds[0].first, li = preds[0].second;
                double d = realized[pi] + compounded[li];
                if (d > realized[n_idx]) {
                    realized[n_idx]   = d;
                    cause_event[n_idx] = pi;
                }
            } else if (!preds.empty()) {
                double maxd = -1e9;
                int best = 0;
                for (int i = 0; i < (int)preds.size(); ++i) {
                    int pi = preds[i].first, li = preds[i].second;
                    double d = realized[pi] + compounded[li];
                    if (d > maxd) { maxd = d; best = i; }
                }
                if (maxd > realized[n_idx]) {
                    realized[n_idx]   = maxd;
                    cause_event[n_idx] = preds[best].first;
                }
            }
        }

        return SimResult{std::move(realized), std::move(compounded), std::move(cause_event)};
    }

    std::vector<SimResult> run_many(const std::vector<int>& seeds) {
        std::vector<SimResult> results;
        results.reserve(seeds.size());
        for (int s : seeds) results.push_back(run(s));
        return results;
    }
};

// ------------------------- Pybind11 -------------------------
PYBIND11_MODULE(_core, m) {
    py::class_<SimulationTreeLink>(m, "SimulationTreeLink")
        .def(py::init<double,int>(), py::arg("minimal_duration"), py::arg("link_type"))
        .def_readwrite("minimal_duration", &SimulationTreeLink::minimal_duration)
        .def_readwrite("link_type", &SimulationTreeLink::link_type);

    py::class_<SimContext>(m, "SimContext")
        .def(py::init([](
            const std::vector<std::pair<std::string,std::tuple<double,double,double>>>& events_raw,
            const std::unordered_map<std::pair<int,int>,std::pair<int,SimulationTreeLink>>& link_map,
            const std::vector<std::pair<int,std::vector<std::pair<int,int>>>>& precedence_list,
            double max_delay
        ) {
            std::vector<std::pair<NodeId,EventPointInTime>> ev;
            ev.reserve(events_raw.size());
            for (auto &e : events_raw)
                ev.emplace_back(e.first,
                    EventPointInTime{std::get<0>(e.second),std::get<1>(e.second),std::get<2>(e.second)});
            return SimContext(std::move(ev), link_map, precedence_list, max_delay);
        }), py::arg("events"), py::arg("link_map"), py::arg("precedence_list"), py::arg("max_delay"))
        .def_readwrite("events", &SimContext::events)
        .def_readwrite("link_map", &SimContext::link_map)
        .def_readwrite("precedence_list", &SimContext::precedence_list)
        .def_readwrite("max_delay", &SimContext::max_delay);

    py::class_<SimResult>(m, "SimResult")
        .def_readonly("realized", &SimResult::realized)
        .def_readonly("delays", &SimResult::delays)
        .def_readonly("cause_event", &SimResult::cause_event);

    py::class_<GenericDelayGenerator>(m, "GenericDelayGenerator")
        .def(py::init<>())
        .def("add_constant", &GenericDelayGenerator::add_constant, py::arg("link_type"), py::arg("factor"))
        .def("add_exponential", &GenericDelayGenerator::add_exponential, py::arg("link_type"), py::arg("lambda"), py::arg("max_scale"))
        .def("set_seed", &GenericDelayGenerator::set_seed, py::arg("seed"));

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<SimContext, GenericDelayGenerator>(), py::arg("context"), py::arg("generator"))
        .def("run", &Simulator::run, py::arg("seed"))
        .def("run_many", &Simulator::run_many, py::arg("seeds"));
}