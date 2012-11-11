using System.Linq;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace DocumentQuery.Core.MultiClassBayesPointMachine
{
    /// <summary>
    /// The multi-class Bayes Point Machine class.
    /// </summary>
    public class Machine : ClassifiedVectorsMachine
    {
        #region Private fields
        
        /// <summary>
        /// The testing model.
        /// </summary>
        private TestModel testModel;

        /// <summary>
        /// The training model
        /// </summary>
        private TrainModel trainModel;

        #endregion

        #region Constructors

        public Machine(int numOfClasses, int numOfFeatures)
            : this(numOfClasses, numOfFeatures, DefaultNoise)
        { }

        public Machine(int numOfClasses, int numOfFeatures, double noise)
            : this(numOfClasses, numOfFeatures, new int[] { }, noise)
        { }

        public Machine(int numOfClasses, int numOfFeatures, int[] featureSelection)
            : this(numOfClasses, numOfFeatures, featureSelection, DefaultNoise)
        { }

        public Machine(int numOfClasses, int numOfFeatures, int[] featureSelection, double noise)
            : base(numOfClasses, numOfFeatures, featureSelection, noise)
        {
            this.trainModel = new TrainModel(numOfClasses, noise);
            this.testModel = new TestModel(numOfClasses, noise);
        }

        #endregion

        #region ClassifiedVectorsMachine implementation

        /// <summary>
        /// Train the machine with training data
        /// </summary>
        /// <param name="filePath">The training data file path</param>
        public override void Train(string filePath)
        {
            this.trainModel.Train(CreateClassifiedDataset(filePath).GetClassifiedVectors());
        }

        /// <summary>
        /// Train the machine with training data in chunks
        /// </summary>
        /// <param name="filePath">The training data file path</param>
        /// <param name="chunkSize">The size of each chunk</param>
        public override void Train(string filePath, int chunkSize)
        {
            foreach (var chunk in CreateClassifiedDataset(filePath).GetClassifiedVectorsInChunks(chunkSize))
            {
                this.trainModel.TrainIncremental(chunk);
            }
        }

        /// <summary>
        /// Use machine to classify test data
        /// </summary>
        /// <param name="filePath">The test data file path</param>
        /// <returns>The test distributions</returns>
        public override Discrete[] Test(string filePath)
        {
            return Test(CreateDataset(filePath).GetDataVectors().Select(v => v.FeatureVector).ToArray());
        }

        /// <summary>
        /// Use machine to classify test data
        /// </summary>
        /// <param name="testData">The test data</param>
        /// <returns>The test distributions</returns>
        public override Discrete[] Test(Vector[] testData)
        {
            return testModel.Test(this.trainModel.GetInferredPosterier(), testData);
        }

        #endregion

    }
}
