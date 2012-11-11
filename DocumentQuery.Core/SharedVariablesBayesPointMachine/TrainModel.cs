using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace DocumentQuery.Core.SharedVariablesBayesPointMachine
{
    /// <summary>
    /// The testing model of the shared-variable multi-class Bayes point machine
    /// </summary>
    internal class TrainModel : VectorsTrainModel
    {
        #region Private fields

        /// <summary>
        /// The variables of all classes
        /// </summary>
        private TrainClass[] classes;

        /// <summary>
        /// The number of features in each vector.
        /// </summary>
        private readonly int numOfFeatures;

        /// <summary>
        /// The shared model.
        /// </summary>
        private Model model;

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numOfChunks">The number of chunks of training data.</param>
        /// <param name="numOfClasses">The range of classes.</param>
        /// <param name="numOfFeatures">The number of features in each vector.</param>
        /// <param name="noise">The noise level.</param>
        public TrainModel(int numOfChunks, int numOfClasses, int numOfFeatures, double noise)
            : base(numOfClasses, noise)
        {
            NumberOfChunks = numOfChunks;
            this.numOfFeatures = numOfFeatures;

            InitModel();
        }

        #endregion

        #region Public

        /// <summary>
        /// The number of chunks of training data.
        /// </summary>
        public int NumberOfChunks { get; private set; }

        /// <summary>
        /// Get the shared variables of each class.
        /// </summary>
        /// <returns>The shared variable list.</returns>
        public SharedVariable<Vector>[] GetWeights()
        {
            return this.classes.Select(c => c.SharedWeight).ToArray();
        }

        /// <summary>
        /// Train the model with training data.
        /// </summary>
        /// <param name="trainData">The traning data.</param>
        /// <param name="chunkNumber">The current chunk number.</param>
        public void Train(IList<Vector>[] trainData, int chunkNumber)
        {
            for (int i = 0; i < this.numOfClasses; i++)
            {
                classes[i].SetObservedData(trainData[i]);
            }

            this.model.InferShared(Engine, chunkNumber);

            foreach (var trainClass in classes)
            {
                trainClass.InferWeight(Engine);
            }
        }

        #endregion

        #region ModelBase implementations

        /// <summary>
        /// Initialize fields and properties.
        /// </summary>
        protected override void InitModel()
        {
            // Initalize classes
            this.classes = new TrainClass[this.numOfClasses].Populate(() => new TrainClass());

            // Produce the initial weights in each class
            for (int i = 0; i < this.numOfClasses; i++)
            {
                this.classes[i].SetWeight(i, this.numOfFeatures);
            }

            // Initialize the model
            this.model = new Model(NumberOfChunks);

            // Initialize sub-models
            foreach (var trainClass in classes)
            {
                trainClass.SetSubModel(model);
            }

            ComputeAndConstrainScores(classes);
        }

        #endregion
    }
}
