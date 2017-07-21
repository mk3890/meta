/**
 * @file lda.cpp
 * @author Chase Geigle
 */

#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cpptoml.h"
#include "meta/index/forward_index.h"
#include "meta/learn/dataset.h"
#include "meta/logging/logger.h"
#include "meta/topics/lda_cvb.h"
#include "meta/topics/lda_gibbs.h"
#include "meta/topics/lda_scvb.h"
#include "meta/topics/parallel_lda_gibbs.h"

using namespace meta;

template <class Model, class Dataset>
int run_lda(Dataset& docs, const cpptoml::table& lda_config)
{
    auto num_iters = *lda_config.get_as<uint64_t>("max-iters");
    Model model{docs, lda_config};
    //model.run(num_iters);
    model.run(23);
    model.save();
    return 0;
}

bool check_parameter(const std::string& file, const cpptoml::table& group,
                     const std::string& param)
{
    if (!group.contains(param))
    {
        std::cerr << "Missing lda configuration parameter " << param << " in "
                  << file << std::endl;
        return false;
    }
    return true;
}

int run_lda(const std::string& config_file)
{
    using namespace meta::topics;
    auto config = cpptoml::parse_file(config_file);

    if (!config->contains("lda"))
    {
        std::cerr << "Missing lda configuration group in " << config_file
                  << std::endl;
        return 1;
    }

    auto lda_group = config->get_table("lda");

    if (!check_parameter(config_file, *lda_group, "alpha")
        || !check_parameter(config_file, *lda_group, "beta")
        || !check_parameter(config_file, *lda_group, "topics")
        || !check_parameter(config_file, *lda_group, "inference")
        || !check_parameter(config_file, *lda_group, "max-iters")
        || !check_parameter(config_file, *lda_group, "model-prefix"))
        return 1;

    auto type = *lda_group->get_as<std::string>("inference");

    auto f_idx = index::make_index<index::forward_index>(*config);
    auto doc_list = f_idx->docs();
    learn::dataset docs{f_idx, doc_list.begin(), doc_list.end()};
    if (type == "gibbs")
    {
        std::cout << "Beginning LDA using serial Gibbs sampling..."
                  << std::endl;
        return run_lda<lda_gibbs>(docs, *lda_group);
    }
    else if (type == "pargibbs")
    {
        std::cout << "Beginning LDA using parallel Gibbs sampling..."
                  << std::endl;
        return run_lda<parallel_lda_gibbs>(docs, *lda_group);
    }
    else if (type == "cvb")
    {
        std::cout << "Beginning LDA using serial collapsed variational bayes..."
                  << std::endl;
        return run_lda<lda_cvb>(docs, *lda_group);
    }
    else if (type == "scvb")
    {
        std::cout
            << "Beginning LDA using stochastic collapsed variational bayes... "
            << std::endl;
        return run_lda<lda_scvb>(docs, *lda_group);
    }
    std::cout
        << "Incorrect method selected: must be gibbs, pargibbs, cvb, or scvb"
        << std::endl;
    return 1;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage:\t" << argv[0] << " config.toml" << std::endl;
        return 1;
    }

    logging::set_cerr_logging();
    return run_lda(argv[1]);
}
