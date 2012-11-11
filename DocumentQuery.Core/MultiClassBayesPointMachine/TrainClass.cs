using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace DocumentQuery.Core.MultiClassBayesPointMachine
{
    /// <summary>
    /// Represents a class of vectors for the training model of 
    /// the mutli-class Bayes Point Machine.
    /// </summary>
    internal class TrainClass : VectorsTrainClass
    {
        #region Public properties

        /// <summary>
        /// The initial priors for the weight vector
        /// </summary>
        public Variable<VectorGaussian> InitialPrior { get; private set; }

        #endregion

        #region Constructors

        /// <summary>
        /// The default constructor
        /// </summary>
        public TrainClass() : base()
        {
            InitialPrior = Variable.New<VectorGaussian>();
            Weight = Variable<Vector>.Random(InitialPrior);
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Set the initial prior
        /// </summary>
        /// <param name="numOfFeatures">The number of features in vector</param>
        /// <param name="classId">The class ID</param>
        public void SetInitialPrior(int numOfFeatures, int classId)
        {
            InitialPrior.ObservedValue = classId == 0 ?
                VectorGaussian.PointMass(Vector.Zero(numOfFeatures))
                : VectorGaussian.FromMeanAndPrecision(Vector.Zero(numOfFeatures), PositiveDefiniteMatrix.Identity(numOfFeatures));
        }

        /// <summary>
        /// Set the initial prior observed value with the previous posterior
        /// to support incrementally training
        /// </summary>
        public void SetInitialPriorIncremental()
        {
            InitialPrior.ObservedValue = InferredPosterior;
        }

        /// <summary>
        /// Performs inference on this class
        /// </summary>
        /// <param name="engine">The inference engine</param>
        public void InferWeight(InferenceEngine engine)
        {
            base.InferredPosterior = engine.Infer<VectorGaussian>(Weight);
        }

        #endregion
    }
}
