using System;

namespace DocumentQuery.Core
{
    /// <summary>
    /// The base class for a Bayes Point Machine model.
    /// </summary>
    public abstract class MachineBase
    {
        #region protected fields

        /// <summary>
        /// The default noise level
        /// </summary>
        protected const double DefaultNoise = 0.1;

        /// <summary>
        /// The total number of selected features in a vector.
        /// </summary>
        protected readonly int numOfFeatures;

        /// <summary>
        /// The selected features for modeling
        /// </summary>
        protected readonly int[] featureSelection;

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numOfFeatures">The total number of features in the data file</param>
        /// <param name="featureSelection">The feature selection</param>
        /// <param name="noise">Noise</param>
        protected MachineBase(int numOfFeatures, int[] featureSelection, double noise)
        {
            this.numOfFeatures = numOfFeatures;
            this.Noise = noise;

            this.featureSelection = new int[featureSelection.Length];
            Array.Copy(featureSelection, this.featureSelection, featureSelection.Length);
        }

        #endregion

        #region Public

        /// <summary>
        /// The noise level
        /// </summary>
        public double Noise { get; private set; }

        #endregion

        #region Protected methods
        
        /// <summary>
        /// Utility method to create unclassified dataset from file path
        /// </summary>
        /// <param name="filePath">The file path</param>
        /// <returns>Unclassified dataset</returns>
        protected UnclassifiedDataset CreateDataset(string filePath)
        {
            return (this.featureSelection.Length == 0)
                       ? new UnclassifiedDataset(filePath, this.numOfFeatures)
                       : new UnclassifiedDataset(filePath, this.numOfFeatures, this.featureSelection);
        }

        /// <summary>
        /// Get the actual number of features in a vector
        /// </summary>
        /// <returns>Number of features</returns>
        protected int GetNumOfReturnFeatures()
        {
            return featureSelection.Length == 0 ? numOfFeatures : featureSelection.Length;
        }

        #endregion

        /// <summary>
        /// Train the machine with training data
        /// </summary>
        /// <param name="filePath">The training data file path</param>
        public abstract void Train(string filePath);

        /// <summary>
        /// Train the machine with training data in chunks
        /// </summary>
        /// <param name="filePath">The training data file path</param>
        /// <param name="chunkSize">The size of each chunk</param>
        public abstract void Train(string filePath, int chunkSize);

    }
}
