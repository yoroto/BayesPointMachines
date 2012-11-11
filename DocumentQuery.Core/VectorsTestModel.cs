using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace DocumentQuery.Core
{
    /// <summary>
    /// Abstract class represents a test model supporting test data in vectors.
    /// </summary>
    public abstract class VectorsTestModel : ModelBase
    {
        #region Private fields

        /// <summary>
        /// The variable of the number of test data vectors
        /// </summary>
        protected Variable<int> numOfVectors;

        /// <summary>
        /// The range of test data vectors
        /// </summary>
        protected Range range;

        /// <summary>
        /// The variable array of feature vectors
        /// </summary>
        protected VariableArray<Vector> featureVectors;

        /// <summary>
        /// The variable array of model output
        /// </summary>
        protected VariableArray<int> modelOutput;

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numOfClasses">The number of classes.</param>
        /// <param name="noise">The noise level</param>
        protected VectorsTestModel(int numOfClasses, double noise) 
            : base(numOfClasses, noise)
        {
            // Initialize the infer engine
            this.Engine = new InferenceEngine();

            // Initialize the variable for number of vectors and the range
            this.numOfVectors = Variable.New<int>();
            this.range = new Range(this.numOfVectors);

            // Initialize the variable array for feature vectors
            this.featureVectors = Variable.Array<Vector>(this.range);

            // Initialize the variable array for test results
            this.modelOutput = Variable.Array<int>(this.range);
        }

        #endregion

        #region Protected methods

        /// <summary>
        /// Compute score for vectors, and constrain the score require to be the maximum 
        /// for the class inferred in model results
        /// </summary>
        protected void ComputeAndConstrainScores(Variable<Vector>[] variableVector)
        {
            using (Variable.ForEach(this.range))
            {
                var score = new Variable<double>[numOfClasses];
                var scorePlusNoise = new Variable<double>[numOfClasses];

                for (int i = 0; i < numOfClasses; i++)
                {
                    score[i] = Variable.InnerProduct(variableVector[i], this.featureVectors[this.range]);
                    scorePlusNoise[i] = Variable.GaussianFromMeanAndPrecision(score[i], Noise);
                }

                this.modelOutput[this.range] = Variable.DiscreteUniform(numOfClasses);

                for (int j = 0; j < numOfClasses; j++)
                {
                    using (Variable.Case(this.modelOutput[this.range], j))
                    {
                        for (int k = 0; k < scorePlusNoise.Length; k++)
                        {
                            if (k != j)
                            {
                                Variable.ConstrainPositive(scorePlusNoise[j] - scorePlusNoise[k]);
                            }
                        }
                    }
                }
            }
        }

        #endregion
    }
}
