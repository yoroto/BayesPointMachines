using MicrosoftResearch.Infer;

namespace DocumentQuery.Core
{
    public abstract class ModelBase
    {
        #region Protected fields

        /// <summary>
        /// Number of classes
        /// </summary>
        protected int numOfClasses;

        #endregion

        #region Public methods

        protected ModelBase(int numOfClasses, double noise)
        {
            this.numOfClasses = numOfClasses;
            Noise = noise;
            Engine = new InferenceEngine();
        }

        #endregion

        #region Public properties

        /// <summary>
        /// The inference engine.
        /// </summary>
        public InferenceEngine Engine { get; set; }

        /// <summary>
        /// The pre-defined noise level
        /// </summary>
        public double Noise { get; private set; }

        #endregion

        protected abstract void InitModel();
    }
}
