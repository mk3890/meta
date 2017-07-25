/**
 * @file topics_test.cpp
 * @author Sean Massung
 */

#include <fstream>

#include "bandit/bandit.h"
#include "create_config.h"
#include "meta/index/forward_index.h"
#include "meta/learn/dataset.h"
#include "meta/topics/lda_cvb.h"
#include "meta/topics/lda_gibbs.h"
#include "meta/topics/lda_scvb.h"
#include "meta/topics/parallel_lda_gibbs.h"

using namespace bandit;
using namespace meta;

namespace
{

template <class TopicModel>
void run_model(const learn::dataset& docs)
{
    const std::string prefix = "meta-lda-test";

    {
        auto cfg = tests::create_config("line");
        cfg->insert("topics", 10);
        cfg->insert("alpha", 0.1);
        cfg->insert("beta", 0.1);
        cfg->insert("max-iters", 3);
        cfg->insert("save-period", std::numeric_limits<uint64_t>::max());
        cfg->insert("model", prefix);

        const double delta = 0.0000001;
        const uint64_t num_topics = 3;
        TopicModel model{docs, *cfg};
        AssertThat(model.num_topics(), Equals(num_topics));
        model.run(3); // only run for three iterations

        // all term probs for all topics should sum to 1
        for (uint64_t topic = 0; topic < model.num_topics(); ++topic)
        {
            double sum = 0.0;
            for (uint64_t term = 0; term < docs.total_features(); ++term)
            {
                sum += model.compute_term_topic_probability(term_id{term},
                                                            topic_id{topic});
            }
            AssertThat(sum, EqualsWithDelta(1.0, delta));
        }

        // all topic probs for all docs should sum to 1
        for (const auto& doc : docs)
        {
            double sum = 0.0;
            for (uint64_t topic = 0; topic < model.num_topics(); ++topic)
                sum += model.compute_doc_topic_probability(doc.id,
                                                           topic_id{topic});
            AssertThat(sum, EqualsWithDelta(1.0, delta));
        }
        model.save();
    }
    AssertThat(filesystem::file_exists(prefix + ".phi.bin"), IsTrue());
    AssertThat(filesystem::file_exists(prefix + ".theta.bin"), IsTrue());
    filesystem::delete_file(prefix + ".phi");
    filesystem::delete_file(prefix + ".theta");
}
}

go_bandit([]() {

    describe("[topics]", [&]() {
        const std::string prefix = "meta-test-lda-model";
        auto config = tests::create_config("line");
        auto idx = index::make_index<index::forward_index>(*config);
        auto doc_list = idx->docs();
        learn::dataset docs{idx, doc_list.begin(), doc_list.end()};

        it("should run LDA with CVB inference",
           [&]() { run_model<topics::lda_cvb>(docs); });

        it("should run LDA with Gibbs sampling inference",
           [&]() { run_model<topics::lda_gibbs>(docs); });

        it("should run LDA with SCVB0 inference",
           [&]() { run_model<topics::lda_scvb>(docs); });

        it("should run LDA with parallel Gibbs inference",
           [&]() { run_model<topics::parallel_lda_gibbs>(docs); });
    });

    filesystem::remove_all("ceeaus");
});
