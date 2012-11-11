using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace DocumentQuery.Core
{
    public class SimpleBayesPointMachine : MachineBase
    {
        #region Private fields

        /// <summary>
        /// The weight vector as a random variable with a broad prior
        /// </summary>
        private Variable<Vector> weight;

        /// <summary>
        /// Indicate whether the machine has been trained with data
        /// </summary>
        private bool isTrained;

        #endregion

        #region Constructors

        public SimpleBayesPointMachine(int numOfFeatures)
            : this(numOfFeatures, DefaultNoise)
        { }

        public SimpleBayesPointMachine(int numOfFeatures, double noise)
            : this(numOfFeatures, new int[] { }, noise)
        { }

        public SimpleBayesPointMachine(int numOfFeatures, int[] featureSelection)
            : this(numOfFeatures, featureSelection, DefaultNoise)
        { }

        public SimpleBayesPointMachine(int numOfFeatures, int[] featureSelection, double noise)
            : base(numOfFeatures, featureSelection, noise)
        {
            this.isTrained = false;

            Engine = new InferenceEngine();
        }

        #endregion

        #region Public properties

        /// <summary>
        /// The inference engine.
        /// </summary>
        public InferenceEngine Engine { get; set; }

        /// <summary>
        /// The posterior distribution.
        /// </summary>
        public VectorGaussian Posterior { get; private set; }

        #endregion

        #region Public methods

        /// <summary>
        /// Tests the testing file with the trained model. 
        /// </summary>
        /// <param name="filePath">The testing file path.</param>
        /// <returns>The prediction of distribution.</returns>
        public Bernoulli[] Test(string filePath)
        {
            return Test(CreateDataset(filePath).GetDataVectors().Select(v => v.FeatureVector).ToArray());
        }

        /// <summary>
        /// Tests the testing file with the trained model. 
        /// </summary>
        /// <param name="testData">The testing data.</param>
        /// <returns>The prediction of distribution.</returns>
        public Bernoulli[] Test(Vector[] testData)
        {
            if (this.Posterior == null)
            {
                return null;
            }

            // Create an array for storing test results
            VariableArray<bool> testResults = Variable.Array<bool>(new Range(testData.Length)).Named("testResults");

            // Create an array to represent the observed training data            
            Range resultRange = testResults.Range.Named("relevance");
            VariableArray<Vector> observedData =
                Variable.Observed(testData, resultRange).Named("observedData");

            // Create Bayes Point Machine using the posterior from training as prior
            testResults[resultRange] =
                Variable.GaussianFromMeanAndVariance(
                    Variable.InnerProduct(Variable.Random(this.Posterior).Named("weight"), observedData[resultRange]).
                        Named("innerProduct"), this.Noise) > 0;

            // Return test results
            return Engine.Infer<Bernoulli[]>(testResults);
        }

        #endregion

        #region Private methods

        /// <summary>
        /// Train the machine with training data.
        /// </summary>
        /// <param name="trainingData">Training data</param>
        private void TrainModel(IList<DataVector> trainingData)
        {
            // Create the array to store classification results
            VariableArray<bool> trainResults =
                Variable.Observed(trainingData.Select(r => r.ClassId == 1).ToArray()).Named("trainResults");

            InitWeight();

            // Create an array to represent the observed training data            
            Range resultRange = trainResults.Range.Named("relevance");
            VariableArray<Vector> observedData =
                Variable.Observed(trainingData.Select(r => r.FeatureVector).ToArray(), resultRange).Named("observedData");

            // Create Bayes Point Machine
            trainResults[resultRange] =
                Variable.GaussianFromMeanAndVariance(
                    Variable.InnerProduct(weight, observedData[resultRange]).Named("innerProduct"), this.Noise) > 0;

            // Using expectation propagation to infer the posterior over test data
            this.Posterior = Engine.Infer<VectorGaussian>(this.weight);
        }

        /// <summary>
        /// Initialize or re-set the weight variable.
        /// </summary>
        private void InitWeight()
        {
            if (isTrained)
            {
                var initialPrior = Variable.New<VectorGaussian>();
                initialPrior.ObservedValue = Posterior;
                this.weight = Variable<Vector>.Random(initialPrior).Named("weight");
            }
            else
            {
                this.weight = Variable.Random(new VectorGaussian(
                        Vector.Zero(GetNumOfReturnFeatures()),
                        PositiveDefiniteMatrix.Identity(GetNumOfReturnFeatures()))
                    ).Named("weight");
            }
        }

        #endregion

        #region MachineBase implementation

        /// <summary>
        /// Train the machine with training data
        /// </summary>
        /// <param name="filePath">The training data file path</param>
        public override void Train(string filePath)
        {
            TrainModel(CreateDataset(filePath).GetDataVectors());

            isTrained = true;
        }

        /// <summary>
        /// Train the machine with training data in chunks
        /// </summary>
        /// <param name="filePath">The training data file path</param>
        /// <param name="chunkSize">The size of each chunk</param>
        public override void Train(string filePath, int chunkSize)
        {
            foreach (var chunk in CreateDataset(filePath).GetDataVectorInChunk(chunkSize))
            {
                TrainModel(chunk);

                isTrained = true;
            }
        }

        #endregion
    }
}
