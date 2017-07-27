/**
 * @file lda_model.cpp
 * @author Chase Geigle
 */

#include <chrono>
#include <iostream>

#include "meta/io/filesystem.h"
#include "meta/topics/lda_model.h"

namespace meta
{
namespace topics
{

lda_model::lda_model(const learn::dataset& docs,
                     const cpptoml::table& lda_config)
    : docs_(docs),
      num_topics_(lda_config.get_as<std::size_t>("topics").value_or(10)),
      alpha_(lda_config.get_as<double>("alpha").value_or(0.1)),
      beta_(lda_config.get_as<double>("beta").value_or(0.1)),
      max_iters_(lda_config.get_as<uint64_t>("max-iters").value_or(0.1)),
      save_period_(lda_config.get_as<uint64_t>("save-period")
                       .value_or(std::numeric_limits<uint64_t>::max())),
      prefix_(
          lda_config.get_as<std::string>("model-prefix").value_or("lda-model")),
      seed_(lda_config.get_as<uint64_t>("seed").value_or(
          std::chrono::system_clock::now().time_since_epoch().count())),
      iters_elapsed_(0),
      converged_(false)
{
    /* nothing */
    std::cout << "num_topics_: " << num_topics_ << std::endl;
    std::cout << "alpha_: " << alpha_ << std::endl;
    std::cout << "beta_: " << beta_ << std::endl;
    std::cout << "max_iters_: " << max_iters_ << std::endl;
    std::cout << "save_period_: " << save_period_ << std::endl;
    std::cout << "prefix_:" << prefix_ << std::endl;
    std::cout << "seed_:" << seed_ << std::endl;
}

void lda_model::save_doc_topic_distributions(std::ostream& stream) const
{
    io::packed::write(stream, docs_.size());
    io::packed::write(stream, num_topics_);

    for (const auto& d : docs_)
    {
        io::packed::write(stream, topic_distrbution(doc_id{d.id}));
    }
}

void lda_model::save_topic_term_distributions(std::ostream& stream) const
{
    io::packed::write(stream, num_topics_);
    io::packed::write(stream, docs_.total_features());

    for (const auto& p : phi_)
    {
        io::packed::write(stream, p);
    }
}

void lda_model::save() const
{
    save_results("final");
}

void lda_model::load()
{
   // Load the state first so we know which file to open
   load_state();

   std::ifstream theta{ prefix_ + "/results-" + std::to_string(iters_elapsed_)
      + ".theta.bin",
      std::ios::binary };
   std::ifstream phi{ prefix_ + "/results-" + std::to_string(iters_elapsed_)
      + ".phi.bin",
      std::ios::binary };

   {
      if (!theta)
      {
         throw std::runtime_error{ prefix_ + "/results-"
            + std::to_string(iters_elapsed_)
            + ".theta.bin" };
      }

      printing::progress doc_progress{
         " > Loading document topic probabilities: ", docs_.size() };
      io::packed::read<std::size_t>(theta);
      io::packed::read<std::size_t>(theta);
      for (doc_id d{ 0 }; d < docs_.size(); ++d)
      {
         if (!theta)
         {
            throw std::runtime_error{
               "document topic stream ended unexpectedly" };
         }

         doc_progress(d);
         theta_[d] = io::packed::read<stats::multinomial<topic_id>>(theta);
      }
   }

   {
      if (!phi)
      {
         throw std::runtime_error{ prefix_ + "/results-"
            + std::to_string(iters_elapsed_)
            + "phi.bin" };
      }

      printing::progress term_progress{
         " > Loading topic term probabilities: ", num_topics_ };

      io::packed::read<std::size_t>(phi);
      io::packed::read<std::size_t>(phi);
      for (topic_id tid{ 0 }; tid < num_topics_; ++tid)
      {
         if (!phi)
         {
            throw std::runtime_error{
               "topic term stream ended unexpectedly" };
         }

         phi_[tid] = io::packed::read<stats::multinomial<term_id>>(phi);
         term_progress(tid);
      }
   }
}

void lda_model::save_results(const std::string& file_name) const
{
    filesystem::make_directory(prefix_);
    std::ofstream theta_file{prefix_ + "/" + file_name + ".theta.bin",
                             std::ios::binary};
    std::ofstream phi_file{prefix_ + "/" + file_name + ".phi.bin",
                           std::ios::binary};

    save_doc_topic_distributions(theta_file);
    save_topic_term_distributions(phi_file);
}

uint64_t lda_model::num_topics() const
{
    return num_topics_;
}

std::size_t lda_model::doc_size(const learn::instance& inst)
{
    using pair_t = std::pair<learn::feature_id, double>;
    auto sum = std::accumulate(
        inst.weights.begin(), inst.weights.end(), 0.0,
        [](std::size_t amt, const pair_t& in) { return in.second + amt; });
    return static_cast<uint64_t>(sum);
}
}
}
