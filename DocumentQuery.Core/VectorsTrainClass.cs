using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace DocumentQuery.Core
{
    /// <summary>
    /// The abstract class represents a given class with a vector variable array in the training model
    /// </summary>
    public abstract class VectorsTrainClass
    {
        #region Public properties

        /// <summary>
        /// Range of vectors
        /// </summary>
        public Range Range { get; private set; }

        /// <summary>
        /// The variable of the number of vectors 
        /// </summary>
        public Variable<int> NumOfVectors { get; private set; }

        /// <summary>
        /// The variable array of feature vectors
        /// </summary>
        public VariableArray<Vector> FeatureVectors { get; private set; }

        /// <summary>
        /// The weight variable vector within a submodel
        /// </summary>
        public Variable<Vector> Weight { get; protected set; }

        /// <summary>
        /// The inferred posteriors
        /// </summary>
        public VectorGaussian InferredPosterior { get; protected set; }

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        protected VectorsTrainClass()
        {
            NumOfVectors = Variable.New<int>();
            Range = new Range(NumOfVectors);
            FeatureVectors = Variable.Array<Vector>(Range);
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Set the observed training data
        /// </summary>
        /// <param name="trainData">Training data</param>
        public void SetObservedData(IList<Vector> trainData)
        {
            NumOfVectors.ObservedValue = trainData.Count;
            FeatureVectors.ObservedValue = trainData.ToArray();
        }

        #endregion
    }
}
