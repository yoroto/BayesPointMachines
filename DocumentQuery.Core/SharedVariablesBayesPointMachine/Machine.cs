using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace DocumentQuery.Core.SharedVariablesBayesPointMachine
{
    /// <summary>
    /// The shared-variable multi-class Bayes Point Machine model.
    /// </summary>
    public class Machine : ClassifiedVectorsMachine
    {
        #region Private fields

        /// <summary>
        /// The default training chunk size.
        /// </summary>
        private const int DefaultTrainChunkSize = 1000;

        /// <summary>
        /// The default testing chunk size.
        /// </summary>
        private const int DefaultTestChunkSize = Int32.MaxValue;

        /// <summary>
        /// The training model
        /// </summary>
        private TrainModel trainModel;

        /// <summary>
        /// The testing model
        /// </summary>
        private TestModel testModel;

        #endregion

        #region Constructors

        public Machine(int numOfClasses, int numOfFeatures, int numOfTrainChunk)
            : this(numOfClasses, numOfFeatures, numOfTrainChunk, DefaultNoise)
        { }

        public Machine(int numOfClasses, int numOfFeatures, int numOfTrainChunk, double noise)
            : this(numOfClasses, numOfFeatures, numOfTrainChunk, new int[] { }, noise)
        { }

        public Machine(int numOfClasses, int numOfFeatures, int numOfTrainChunk, int[] featureSelection)
            : this(numOfClasses, numOfFeatures, numOfTrainChunk, featureSelection, DefaultNoise)
        { }

        public Machine(int numOfClasses, int numOfFeatures, int numOfTrainChunk, int[] featureSelection, double noise)
            : base(numOfClasses, numOfFeatures, featureSelection, noise)
        {
            this.trainModel = new TrainModel(numOfTrainChunk, numOfClasses, GetNumOfReturnFeatures(), noise);
            this.testModel = new TestModel(1, numOfClasses, noise, this.trainModel.GetWeights());
        }

        #endregion

        #region ClassifiedVectorsMachine implementations

        /// <summary>
        /// Train the machine with training data
        /// </summary>
        /// <param name="filePath">The training data file path</param>
        public override void Train(string filePath)
        {
            Train(filePath, DefaultTrainChunkSize);
        }

        /// <summary>
        /// Train the machine with training data in chunks
        /// </summary>
        /// <param name="filePath">The training data file path</param>
        /// <param name="chunkSize">The size of each chunk</param>
        public override void Train(string filePath, int chunkSize)
        {
            int count = 0;

            foreach (var chunk in CreateClassifiedDataset(filePath).GetClassifiedVectorsInChunks(chunkSize))
            {
                trainModel.Train(chunk, count);
                if (++count == trainModel.NumberOfChunks)
                {
                    break;
                }
            }
        }

        /// <summary>
        /// Use machine to classify test data
        /// </summary>
        /// <param name="filePath">The test data file path</param>
        /// <returns>The test distributions</returns>
        public override Discrete[] Test(string filePath)
        {
            return Test(CreateDataset(filePath).GetDataVectorInChunk(DefaultTestChunkSize).First().Select(v => v.FeatureVector).ToArray());
        }

        /// <summary>
        /// Use machine to classify test data
        /// </summary>
        /// <param name="testData">The test data</param>
        /// <returns>The test distributions</returns>
        public override Discrete[] Test(Vector[] testData)
        {
            return testModel.Test(testData, 0);
        }

        #endregion
    }
}
