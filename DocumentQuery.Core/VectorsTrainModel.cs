using MicrosoftResearch.Infer.Models;

namespace DocumentQuery.Core
{
    /// <summary>
    /// Abstract class represents a training model support data in classified vectors.
    /// </summary>
    public abstract class VectorsTrainModel : ModelBase
    {
        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numOfClasses">Number of classes</param>
        /// <param name="noise">Noise level</param>
        public VectorsTrainModel(int numOfClasses, double noise)
            : base(numOfClasses, noise)
        { }

        #endregion

        #region Protected methods

        /// <summary>
        /// Compute and restrian score values across classes.
        /// </summary>
        /// <param name="classes">All class data variables.</param>
        protected void ComputeAndConstrainScores(VectorsTrainClass[] classes)
        {
            for (int i = 0; i < numOfClasses; i++)
            {
                using (Variable.ForEach(classes[i].Range))
                {
                    var score = new Variable<double>[numOfClasses];
                    var scorePlusNoise = new Variable<double>[numOfClasses];
                    for (int j = 0; j < numOfClasses; j++)
                    {
                        score[j] = Variable.InnerProduct(classes[j].Weight, classes[i].FeatureVectors[classes[i].Range]);
                        scorePlusNoise[j] = Variable.GaussianFromMeanAndPrecision(score[j], Noise);
                    }

                    for (int j = 0; j < numOfClasses; j++)
                    {
                        if (i != j)
                        {
                            Variable.ConstrainPositive(scorePlusNoise[i] - scorePlusNoise[j]);
                        }
                    }
                }
            }
        }

        #endregion
    }
}
