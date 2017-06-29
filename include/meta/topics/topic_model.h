/**
 * @file topic_model.h
 * @author Matt Kelly
 *
 * All files in META are dual-licensed under the MIT and NCSA licenses. For more
 * details, consult the file LICENSE.mit and LICENSE.ncsa in the root of the
 * project.
 */

#ifndef META_TOPICS_TOPICS_H_
#define META_TOPICS_TOPICS_H_

#include <istream>
#include <vector>

#include "cpptoml.h"
#include "meta/config.h"
#include "meta/index/forward_index.h"
#include "meta/util/fixed_heap.h"
#include "meta/util/string_view.h"
#include "meta/util/aligned_allocator.h"

namespace meta
{
namespace topics
{

struct term
{
	std::size_t tid;
	std::string text;
	double probability;
};

struct topic
{
	std::size_t tid;
	double probability;
};

/**
* A read-only model for accessing topic models.
*/
class topic_model
{
	public:

		/**
		* Load topic models from files.
		*
		* @param config The config to read from
		* @param theta The stream to read the vocabulary from
		* @param phi The stream to read the vectors from
		*/
		topic_model(const cpptoml::table& config, std::istream& theta, std::istream& phi);

		/**
		* @param topic_id The topic to use  
		* @param k The number of words to return
		* @return the top k most probable words in the topic
		*/
		std::vector<term> top_k(std::size_t topic_id,
										std::size_t k = 10) const;

		/**
		* @param doc_id The document we are concerned with
		* @return The probability of each of k topics for the 
		* given document
		*/
		std::vector<topic> topic_distribution(std::size_t doc) const;

		/**
		* @param topic_id The topic we are concerned with
		* @param term The term we are concerned with
		* @return The probability of the term for the given topic
		*/
		term term_probability(std::size_t topic_id,
									 util::string_view term) const;

		/**
		* @param doc The document we are concerned with
		* @param topic_id The topic we are concerned with
		* @return The probability for the given topic
		*/
		// TODO: Would the probability of the topic over all documents
		// be useful? Duplicate this function but remove the doc param?
		topic topic_probability(std::size_t doc,
										std::size_t topic_id) const;

		const std::size_t& num_topics() const;
	
	private:

		/**
		* The forward index of the dataset
		*/
		std::shared_ptr<index::cached_index<index::forward_index, caching::no_evict_cache>> index_;

		/**
		* The number of topics.
		*/
		const std::size_t num_topics_;

		/**
		* The number of total unique words.
		*/
		const std::size_t num_words_;

		/**
		* The number of documents.
		*/
		const std::size_t num_docs_;

		/**
		* The term probabilities by topic
		*/
		std::vector<util::aligned_vector<double>> topic_term_probabilities_;

		/**
		* The term probabilities by topic
		*/
		std::vector<util::aligned_vector<double>> doc_topic_probabilities_;
};

class topic_model_exception : public std::runtime_error
{
	public:
		using std::runtime_error::runtime_error;
};

topic_model load_topic_model(const cpptoml::table& config);

}
}

#endif
