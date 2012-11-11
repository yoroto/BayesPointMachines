using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace DocumentQuery.Core
{
    /// <summary>
    /// Abstract class represents the models allowing classified vectors as
    /// training data.
    /// </summary>
    public abstract class ClassifiedVectorsMachine : MachineBase
    {
        #region protected fields

        /// <summary>
        /// The number of classes
        /// </summary>
        protected readonly int numOfClasses;

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numOfClasses">The number of classes</param>
        /// <param name="numOfFeatures">The total number of features in the data file</param>
        /// <param name="featureSelection">The feature selection</param>
        /// <param name="noise">Noise</param>
        protected ClassifiedVectorsMachine(int numOfClasses, int numOfFeatures, int[] featureSelection, double noise)
            : base(numOfFeatures, featureSelection, noise)
        {
            this.numOfClasses = numOfClasses;
        }

        #endregion
        
        #region Protected methods

        /// <summary>
        /// Utility method to create classified dataset from file path
        /// </summary>
        /// <param name="filePath">The file path</param>
        /// <returns>Classified dataset</returns>
        protected ClassifiedDataset CreateClassifiedDataset(string filePath)
        {
            return (this.featureSelection.Length == 0)
                ? new ClassifiedDataset(filePath, this.numOfFeatures, this.numOfClasses)
                : new ClassifiedDataset(filePath, this.numOfFeatures, this.featureSelection, this.numOfClasses);
        }
        
        #endregion

        /// <summary>
        /// Use machine to classify test data
        /// </summary>
        /// <param name="filePath">The test data file path</param>
        /// <returns>The test distributions</returns>
        public abstract Discrete[] Test(string filePath);

        /// <summary>
        /// Use machine to classify test data
        /// </summary>
        /// <param name="testData">The test data</param>
        /// <returns>The test distributions</returns>
        public abstract Discrete[] Test(Vector[] testData);
    }
}
