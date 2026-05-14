## MatchFlow: A Library for Composing Matching Workflows

Maintained by MadMatcher LLC. See [PROVENANCE.md](PROVENANCE.md) for provenance details.

MatchFlow is an open-source library for the matching step of entity matching (EM).

Matching two tables A and B is typically done in two stages. First, a _blocking step_ uses fast heuristics to identify candidate tuple pairs that are likely to match, often using tools such as Sparkly or Delex. Second, a _matching step_ examines these candidate pairs more carefully—using more expensive computation—to determine whether they are true matches.

MatchFlow focuses on this second stage. It provides a machine-learning–based solution for matching and is distinguished by the following features:

- _Supervised ML matchers._ MatchFlow uses supervised learning: users train an ML classifier on labeled tuple pairs and then apply the trained classifier to the blocking output to classify each pair as match or no-match. We refer to this classifier as a _matcher_.
- _Composable matching workflows._ In practice, users often experiment with multiple workflows when building a matcher. MatchFlow provides a set of core functions that can be flexibly composed to create a wide variety of workflows. For example, a user may generate a candidate set of features, refine this set into a final feature set, create training data by labeling tuple pairs, convert labeled pairs into feature vectors, train and evaluate different matchers, and so on.
- _Support for creating high-quality training data._ Producing good labeled data is one of the hardest parts of entity matching. MatchFlow addresses this challenge using active learning, allowing users to focus labeling effort where it is most valuable.
- _Scalability to massive blocking outputs._ Blocking outputs often contain hundreds of millions or even billions of tuple pairs. MatchFlow is designed to scale to these sizes, building on over a decade of research and development from the [Magellan project](https://anhaidgroup.github.io/magellan) at the University of Wisconsin–Madison.
- _Proven in real-world use._ Variants of MatchFlow have been implemented in industry settings and used by hundreds of users.

### How MatchFlow Works

Let C be the output of blocking on tables A and B (using a blocking solution such as [Sparkly](https://github.com/MadMatcher/sparkly) or [Delex](https://github.com/MadMatcher/delex)). Let D be the training data—a set of tuple pairs labeled as match or no-match.

Using MatchFlow, users typically execute the following workflows:

1. _Feature creation._ Run a workflow that examines the schemas and data of tables A and B to generate a set of candidate features. These features typically involve similarity functions commonly used in entity matching. Users may inspect and edit this set to produce a final feature set F.
2. _Feature vector construction._ Convert each tuple pair in the training data D into a feature vector using F. Run a separate workflow to convert each tuple pair in the blocking output C into a feature vector.
3. _Matcher training._ Use the feature vectors derived from D to train a matcher M, which is a classification model such as Random Forest or XGBoost.
4. _Matcher application._ Apply the trained matcher M to the feature vectors from C to predict match or no-match for each candidate pair.

If training data D is not available, MatchFlow provides a workflow to create it using active learning. This workflow iteratively selects a small number of informative tuple pairs from the blocking output C (e.g., a few hundred pairs) for the user to label, progressively improving the matcher with minimal labeling effort.

A key challenge in these workflows is scalability. Blocking outputs often contain hundreds of millions of tuple pairs. This raises many problems. For example, even generating feature vectors is difficult, as it is not trivially parallelizable.

As another example, active learning cannot be applied directly to the blocking output C due to its size and must instead operate on a carefully constructed sample S of C, which is itself non-trivial to obtain. MatchFlow provides effective solutions to address these scalability challenges, enabling matching workflows to operate at real-world scale.

### Runtime Environments

The above workflows are created by stitching together MatchFlow functions and Python code in Python scripts. You can run these scripts in three runtime environments:

- _Pandas on a single machine:_ Use this if you have a relatively small amount of data, or just want to experiment with MatchFlow.
- _Spark on a single machine:_ Use this if you have a relatively small amount of data, or just want to experiment with MatchFlow, or if you want to test your Spark scripts before running them on a cluster.
- _Spark on a cluster of machines:_ Use this if you have a large amount of data, such as 5M+ tuples or more in Tables A and B.

### Additional Details

- The core functions of MatchFlow include _create_features_, _featurize_, _down_sample_, _create_seeds_, _label_data_, _label_pairs_, _train_matcher_, _apply_matcher_.
- Features use well-known similarity measures such as _TF/IDF_, _Jaccard_, _Overlap_, _Cosine_, _etc._, and well-known tokenizers such as _word-level tokenization_ and _qgrams_.
- Labelers are objects that allow users to label using command-line interface and Web browser. If all matches are known, you can use them to simulate labeling (using the Gold Labeler). This facilitates development, debugging, and computing matching accuracy.
- Matchers can use classification models from Scikit Learn and PySpark MLlib.
- Features, labelers, and matchers can be customized and new types can be easily added.

### Case Studies and Performance Statistics

We have used the functions in MatchFlow or earlier variants in many real-world applications in domain science and industry. We will report more details here in the near future.

### Installation

See instructions to install MatchFlow on a [single machine](https://github.com/MadMatcher/MatchFlow/blob/main/docs/installation-guides/install-single-machine.md) or a [cloud-based cluster](https://github.com/MadMatcher/MatchFlow/blob/main/docs/installation-guides/install-cloud-based-cluster.md).

### How to Use

See the [MatchFlow technical guide](https://github.com/MadMatcher/MatchFlow/blob/main/docs/matchflow-technical-guide.md) for an in-depth look into the format, working, and usage of MatchFlow functions. See this page for the [Python scripts of examples of EM workflows](https://github.com/MadMatcher/MatchFlow/blob/main/docs/workflow-examples.md) created with MatchFlow. We recommend reading the technical guide before examining the Python scripts. You may also want to take a quick look at [this brief note about Spark](https://github.com/MadMatcher/MatchFlow/blob/main/docs/spark-note.md).

### Further Pointers

See [API documentation](https://madmatcher.github.io/MatchFlow). For questions / comments, contact [MadMatcher](mailto:dev@hellomadmatcher.com).
