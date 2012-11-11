using System;
using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer.Maths;

namespace DocumentQuery.Core
{
    /// <summary>
    /// Base abstract class for dataset.
    /// </summary>
    public abstract class Dataset
    {
        #region Private and protected fields

        private static readonly char[] RecordDelimiters = new char[] { ' ' };
        private static readonly char[] QueryIdDelimiters = new char[] { ':' };
        private static readonly char[] DocumentIdDelimiters = new char[] { '=' };

        /// <summary>
        /// Number of features in data record.
        /// </summary>
        protected readonly int numOfFeatures;

        /// <summary>
        /// The file path of the training data file.
        /// </summary>
        protected readonly string filepath;

        /// <summary>
        /// The inverted index of the feature selection.
        /// </summary>
        protected readonly Dictionary<int, int> selectedFeatures;

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="filepath">The file path of training data.</param>
        /// <param name="numOfFeatures">The number of features in a data record.</param>
        internal Dataset(string filepath, int numOfFeatures)
        {
            this.filepath = filepath;
            this.numOfFeatures = numOfFeatures;
            this.selectedFeatures = new Dictionary<int, int>();

            this.SkipParsingErrors = true;
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="filepath">The file path of training data.</param>
        /// <param name="numOfFeatures">The number of features in a data record.</param>
        /// <param name="selectedFeatures">The selected features.</param>
        internal Dataset(string filepath, int numOfFeatures, int[] selectedFeatures)
            : this(filepath, numOfFeatures)
        {
            if (selectedFeatures.Any(f => f > numOfFeatures))
            {
                throw new IndexOutOfRangeException("There are features in the selection out of the defined feature range.");
            }

            this.selectedFeatures = selectedFeatures.ToIndexDictionary();
        }

        #endregion

        #region Public properties

        /// <summary>
        /// Indicate whether to skip parsing errors on data records.
        /// If set to true, which is the default, it will ignore format errors and continue.
        /// If set to false, it would throw an exception on format errors.
        /// </summary>
        public bool SkipParsingErrors { get; set; }

        #endregion

        #region Internal methods

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="record">A data record string.</param>
        /// <returns>A data vector.</returns>
        internal DataVector CreateDataVector(string record)
        {
            string[] tokens = record.Split(RecordDelimiters);

            if (tokens.Length < numOfFeatures + 3)
            {
                throw new DatasetFormatException(
                    string.Format(
                        "The expected number of items in the record is {0}, but {1} records have been found: '{2}'",
                        numOfFeatures + 3, tokens.Length, record));
            }

            var cv = new DataVector
            {
                ClassId = ParseClassId(tokens[0]),
                QueryId = ParseQueryId(tokens[1]),
                FeatureVector = Vector.FromArray(
                   this.selectedFeatures.Count > 0 ?
                       ParseSelectedFeatures(tokens.Skip(2).Take(numOfFeatures).ToArray())
                       : ParseFeatures(tokens.Skip(2).Take(numOfFeatures).ToArray())),
                DocumentId = ParseDocumentId(tokens.Skip(numOfFeatures + 2).ToArray())
            };

            return cv;
        }

        /// <summary>
        /// Parse a class ID
        /// </summary>
        /// <param name="classIdToken">Token text.</param>
        /// <returns>Class ID</returns>
        internal static int ParseClassId(string classIdToken)
        {
            int classId;
            if (!int.TryParse(classIdToken, out classId))
            {
                throw new DatasetFormatException(
                    string.Format("The format of class ID ({0}) is wrong.", classIdToken));
            }

            return classId;
        }

        /// <summary>
        /// Parse a query ID
        /// </summary>
        /// <param name="queryIdToken">Token text.</param>
        /// <returns>Query ID</returns>
        internal static string ParseQueryId(string queryIdToken)
        {
            string[] tokens = queryIdToken.Split(QueryIdDelimiters, 2);
            if (tokens.Length != 2 || tokens[0] != "qid" || tokens[1] == string.Empty)
            {
                throw new DatasetFormatException(
                    string.Format("The format of query ID ({0}) is wrong.", queryIdToken));
            }

            return tokens[1];
        }

        /// <summary>
        /// Parse a document ID.
        /// </summary>
        /// <param name="restTokens">Token text.</param>
        /// <returns>Document ID</returns>
        internal static string ParseDocumentId(string[] restTokens)
        {
            if (restTokens.Length <= 2)
            {
                throw new DatasetFormatException(
                    string.Format("Failed to parse the document ID record since the number of token is {0}",
                                  restTokens.Length));
            }
            else if (restTokens[1] != "=" || restTokens[0] != "#docid" || restTokens[2] == string.Empty)
            {
                throw new DatasetFormatException(
                    string.Format("Failed to parse the document ID record: '{0} {1} {2}'",
                                  restTokens[0], restTokens[1], restTokens[2]));
            }

            return restTokens[2];
        }

        /// <summary>
        /// Parse a list of features, and only return the selected features.
        /// </summary>
        /// <param name="featureTokens">Token lists.</param>
        /// <returns>A selected list of feature values.</returns>
        internal double[] ParseSelectedFeatures(string[] featureTokens)
        {
            double[] features = new double[selectedFeatures.Count];

            for (int i = 0; i < numOfFeatures; i++)
            {
                if (this.selectedFeatures.ContainsKey(i + 1))
                {
                    features[this.selectedFeatures[i + 1]] = ParseFeature(featureTokens[i], i + 1);
                }
            }

            return features;
        }

        /// <summary>
        /// Parse a list of features.
        /// </summary>
        /// <param name="featureTokens">Token lists.</param>
        /// <returns>A list of feature values.</returns>
        internal double[] ParseFeatures(string[] featureTokens)
        {
            double[] features = new double[numOfFeatures];

            for (int i = 0; i < numOfFeatures; i++)
            {
                features[i] = ParseFeature(featureTokens[i], i + 1);
            }

            return features;
        }

        /// <summary>
        /// Parse a single feature.
        /// </summary>
        /// <param name="token">Token text.</param>
        /// <param name="index">Feature index.</param>
        /// <returns>Feature value.</returns>
        internal static double ParseFeature(string token, int index)
        {
            string[] tf = token.Split(':');
            int featureNum;
            double featureValue;
            if (tf.Length != 2 || !int.TryParse(tf[0], out featureNum) || featureNum != index || !double.TryParse(tf[1], out featureValue))
            {
                throw new DatasetFormatException(
                    string.Format(
                        "Faild to parse the feature at position {0}: '{1}'",
                            index, token));
            }

            return featureValue;
        }

        #endregion

    }
}
