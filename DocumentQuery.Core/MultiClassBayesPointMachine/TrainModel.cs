using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace DocumentQuery.Core.MultiClassBayesPointMachine
{
    /// <summary>
    /// The training model of the multi-class Bayes point machine
    /// </summary>
    internal sealed class TrainModel : VectorsTrainModel
    {
        #region Private fields

        /// <summary>
        /// All classes
        /// </summary>
        private TrainClass[] classes;

        /// <summary>
        /// Indicate whether the machine has been trained already
        /// </summary>
        private bool isTrained;

        #endregion

        #region Constructors
        
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numOfClasses">Number of classes</param>
        /// <param name="noise">Noise level</param>
        public TrainModel(int numOfClasses, double noise)
            :base(numOfClasses, noise)
        {
            isTrained = false;

            InitModel();
        }

        #endregion
        
        #region Public methods

        /// <summary>
        /// Get a copy of the inferred posteriors of the training model.
        /// </summary>
        /// <returns>A copy of the inferred posteriors</returns>
        public VectorGaussian[] GetInferredPosterier()
        {
            return this.classes.Select(c => c.InferredPosterior).ToArray();
        }

        /// <summary>
        /// Trains this Bayes point machine
        /// </summary>
        /// <param name="trainData"></param>
        public void Train(IList<Vector>[] trainData)
        {
            int numOfFeatures = trainData[0][0].Count;

            for (int i = 0; i < this.numOfClasses; i++)
            {
                classes[i].SetInitialPrior(numOfFeatures, i);
                classes[i].SetObservedData(trainData[i]);
            }

            for (int i = 0; i < this.numOfClasses; i++)
            {
                classes[i].InferWeight(Engine);
            }
            isTrained = true;
        }

        /// <summary>
        /// Incrementally trains this Bayes point machine.
        /// </summary>
        /// <param name="trainData"></param>
        public void TrainIncremental(IList<Vector>[] trainData)
        {
            if (isTrained)
            {
                for (int i = 0; i < numOfClasses; i++)
                {
                    classes[i].SetInitialPriorIncremental();
                    classes[i].SetObservedData(trainData[i]);
                }

                for (int i = 0; i < numOfClasses; i++)
                {
                    classes[i].InferWeight(Engine);
                }
            }
            else
            {
                Train(trainData);
            }
        }

        #endregion

        #region ModelBase implementations

        /// <summary>
        /// Initialize all the fields and properties.
        /// </summary>
        protected override void InitModel()
        {
            // Initialize train classes
            this.classes = new TrainClass[this.numOfClasses].Populate(() => new TrainClass());

            ComputeAndConstrainScores(classes);
        }

        #endregion
    }
}
