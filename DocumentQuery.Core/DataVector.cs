using MicrosoftResearch.Infer.Maths;

namespace DocumentQuery.Core
{
    /// <summary>
    /// A structure to represents a data record.
    /// </summary>
    public struct DataVector
    {
        /// <summary>
        /// Class ID
        /// </summary>
        public int ClassId;

        /// <summary>
        /// Query ID
        /// </summary>
        public string QueryId;

        /// <summary>
        /// Feature vector
        /// </summary>
        public Vector FeatureVector;

        /// <summary>
        /// Document ID
        /// </summary>
        public string DocumentId;
    }
}
