using System.Linq;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace DocumentQuery.Core.MultiClassBayesPointMachine
{
    /// <summary>
    /// The testing model of the multi-class Bayes point machine
    /// </summary>
    internal sealed class TestModel : VectorsTestModel
    {
        #region Private fields

        /// <summary>
        /// The test variables of each class
        /// </summary>
        private TestClass[] classes;

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numOfClasses">Number of classes</param>
        /// <param name="noise">Noise level</param>
        public TestModel(int numOfClasses, double noise)
            : base(numOfClasses, noise)
        {
            InitModel();
        }

        #endregion
        
        #region Public methods

        /// <summary>
        /// Test the test data.
        /// </summary>
        /// <param name="priorValue">Prior distribution from training.</param>
        /// <param name="testData">The test data.</param>
        /// <returns>The prediction.</returns>
        public Discrete[] Test(VectorGaussian[] priorValue, Vector[] testData)
        {
            // Set the prior of all classes
            for (int i = 0; i < numOfClasses; i++)
            {
                this.classes[i].Prior.ObservedValue = priorValue[i];
            }

            // Set the observed test data
            this.numOfVectors.ObservedValue = testData.Length;
            this.featureVectors.ObservedValue = testData;

            // Infer the test model outputs
            return Distribution.ToArray<Discrete[]>(this.Engine.Infer(this.modelOutput));
        }

        #endregion

        #region ModelBase implementation

        /// <summary>
        /// Initialize fields and properties of the test model
        /// </summary>
        protected override void InitModel()
        {
            // Initialize all the weight and its prior for all classes
            this.classes = new TestClass[numOfClasses].Populate(() => new TestClass());

            // Constrain scores
            ComputeAndConstrainScores(classes.Select(c => c.Weight).ToArray());;
        }

        #endregion

        /// <summary>
        /// Test variables for each class
        /// </summary>
        private class TestClass
        {
            /// <summary>
            /// The variable vector of weight.
            /// </summary>
            public Variable<Vector> Weight { get; private set; }

            /// <summary>
            /// The variable vector of prior distribution.
            /// </summary>
            public Variable<VectorGaussian> Prior { get; private set; }

            /// <summary>
            /// Constructor
            /// </summary>
            public TestClass()
            {
                Prior = Variable.New<VectorGaussian>();

                Weight = Variable<Vector>.Random(Prior);
            }
        }
    }
}
