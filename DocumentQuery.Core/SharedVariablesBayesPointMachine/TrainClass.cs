using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace DocumentQuery.Core.SharedVariablesBayesPointMachine
{
    /// <summary>
    /// Represents a class of vectors for the training model of 
    /// the share-variable mutli-class Bayes Point Machine.
    /// </summary>
    internal class TrainClass : VectorsTrainClass
    {
        #region Public properties

        /// <summary>
        /// The weight vector
        /// </summary>
        public SharedVariable<Vector> SharedWeight { get; set; }

        #endregion

        #region Public methods

        /// <summary>
        /// Set the weight shared variables.
        /// </summary>
        /// <param name="index">The class ID.</param>
        /// <param name="numOfFeatures">The number of features in the class.</param>
        public void SetWeight(int index, int numOfFeatures)
        {
            SharedWeight = (index == 0)
                    ? SharedVariable<Vector>.Random(VectorGaussian.PointMass(Vector.Zero(numOfFeatures)))
                    : SharedVariable<Vector>.Random(
                        VectorGaussian.FromMeanAndPrecision(
                        Vector.Zero(numOfFeatures),
                        PositiveDefiniteMatrix.Identity(numOfFeatures)));
        }

        /// <summary>
        /// Initalize the sub model weight variable.
        /// </summary>
        /// <param name="model">The shared model.</param>
        public void SetSubModel(Model model)
        {
            Weight = SharedWeight.GetCopyFor(model);
        }

        /// <summary>
        /// Performs inference on this class
        /// </summary>
        /// <param name="engine">The inference engine</param>
        public void InferWeight(InferenceEngine engine)
        {
            InferredPosterior = SharedWeight.Marginal<VectorGaussian>();
        }

        #endregion
    }
}
