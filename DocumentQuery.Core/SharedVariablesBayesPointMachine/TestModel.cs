using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace DocumentQuery.Core.SharedVariablesBayesPointMachine
{
    /// <summary>
    /// The testing model of the shared-variable multi-class Bayes Point Machine.
    /// </summary>
    internal sealed class TestModel : VectorsTestModel
    {
        #region Private fields

        /// <summary>
        /// The model for the shared variables
        /// </summary>
        private Model model;

        /// <summary>
        /// The shared variable of weights
        /// </summary>
        private SharedVariable<Vector>[] weights;

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numOfChunks">The number of chunks.</param>
        /// <param name="numOfClasses">The range of classes.</param>
        /// <param name="noise">The noise level.</param>
        /// <param name="weights">The weight shared variable the from training model.</param>
        public TestModel(int numOfChunks, int numOfClasses, double noise, SharedVariable<Vector>[] weights)
            : base(numOfClasses, noise)
        {
            NumberOfChunks = numOfChunks;
            this.weights = weights;

            InitModel();
        }

        #endregion

        #region Public methods

        /// <summary>
        /// The number of chunks in testing.
        /// </summary>
        public int NumberOfChunks { get; private set; }

        /// <summary>
        /// Test the testing data.
        /// </summary>
        /// <param name="testData">The testing data.</param>
        /// <param name="chunkNumber">The current chunk number.</param>
        /// <returns>The prediction of tests.</returns>
        public Discrete[] Test(Vector[] testData, int chunkNumber)
        {
            for (int i = 0; i < this.numOfClasses; i++)
            {
                this.weights[i].SetInput(this.model, chunkNumber);
            }
            
            // Set the observed test data
            this.numOfVectors.ObservedValue = testData.Length;
            this.featureVectors.ObservedValue = testData;

            return Distribution.ToArray<Discrete[]>(this.Engine.Infer(this.modelOutput));
        }

        #endregion

        #region ModelBase implementation

        /// <summary>
        /// Initialize fields and properties of the test model
        /// </summary>
        protected override void InitModel()
        {
            // Initialize the model
            this.model = new Model(NumberOfChunks);

            Variable<Vector>[] subModel = new Variable<Vector>[this.numOfClasses];
            for (int i = 0; i < this.numOfClasses; i++)
            {
                // Get a copy of the shared weight vector variable for the submodel
                subModel[i] = this.weights[i].GetCopyFor(model);
            }

            // Constrain scores
            ComputeAndConstrainScores(subModel);
        }

        #endregion
    }
}
