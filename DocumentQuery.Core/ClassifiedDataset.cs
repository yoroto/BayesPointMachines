using System.Collections.Generic;
using System.IO;
using MicrosoftResearch.Infer.Maths;

namespace DocumentQuery.Core
{
    /// <summary>
    /// The dataset loader to output classified vector lists.
    /// </summary>
    public class ClassifiedDataset : Dataset
    {
        #region Private fields
        
        /// <summary>
        /// Number of classes
        /// </summary>
        private readonly int numOfClasses;
        
        #endregion

        #region Public properties

        /// <summary>
        /// Indicate whether to skip the record and continue if the class ID
        /// of that record is out of defined range. It would throw an exception,
        /// if set to false. By default, it is true.
        /// </summary>
        public bool SkipClassValueOutOfRange { get; set; }

        #endregion

        #region Public methods

        /// <summary>
        /// Return the classified vectors. The index of the array is the class ID,
        /// and the content is a list of vectors who are in that class.
        /// </summary>
        /// <returns>Classified vectors</returns>
        public IList<Vector>[] GetClassifiedVectors()
        {
            IList<Vector>[] vectors = new IList<Vector>[numOfClasses].Populate(() => new List<Vector>());

            using (var sr = new StreamReader(this.filepath))
            {
                while (sr.Peek() >= 0)
                {
                    DataVector dataVector;

                    try
                    {
                        dataVector = CreateDataVector(sr.ReadLine());
                    }
                    catch (DatasetFormatException e)
                    {
                        if (SkipParsingErrors)
                        {
                            continue;
                        }
                        else
                        {
                            throw e;
                        }
                    }

                    if (!CheckClassId(dataVector.ClassId))
                    {
                        continue;
                    }

                    vectors[dataVector.ClassId].Add(dataVector.FeatureVector);
                }
            }

            return vectors;
        }

        /// <summary>
        /// Return the classified vectors in chunks as a Emurator.
        /// </summary>
        /// <returns>Classified vectors</returns>
        public IEnumerable<IList<Vector>[]> GetClassifiedVectorsInChunks(int chunkSize)
        {
            using (var sr = new StreamReader(this.filepath))
            {
                bool fileEnded = false;
                while (!fileEnded)
                {
                    IList<Vector>[] vectors = new IList<Vector>[numOfClasses].Populate(() => new List<Vector>());

                    int i = 0;
                    for (i = 0; i < chunkSize; i++)
                    {
                        if (sr.Peek() < 0)
                        {
                            fileEnded = true;
                            break;
                        }

                        DataVector dataVector;

                        try
                        {
                            dataVector = CreateDataVector(sr.ReadLine());
                        }
                        catch (DatasetFormatException e)
                        {
                            if (SkipParsingErrors)
                            {
                                continue;
                            }
                            else
                            {
                                throw e;
                            }
                        }

                        if (!CheckClassId(dataVector.ClassId))
                        {
                            continue;
                        }

                        vectors[dataVector.ClassId].Add(dataVector.FeatureVector);
                    }

                    if (i > 0)
                    {
                        yield return vectors;
                    }
                    else
                    {
                        yield break;
                    }
                }
            }
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="filepath">The training file path</param>
        /// <param name="numOfFeatures">Number of features in data record</param>
        /// <param name="numOfClasses">Number of classes</param>
        internal ClassifiedDataset(string filepath, int numOfFeatures, int numOfClasses)
            : base(filepath, numOfFeatures)
        {
            this.numOfClasses = numOfClasses;

            this.SkipClassValueOutOfRange = true;
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="filepath">The training file path</param>
        /// <param name="numOfFeatures">Number of features in data record</param>
        /// <param name="selectedFeatures">The selected features. All features would be considered if this is empty.</param>
        /// <param name="numOfClasses">Number of classes</param>
        internal ClassifiedDataset(string filepath, int numOfFeatures, int[] selectedFeatures, int numOfClasses)
            : base(filepath, numOfFeatures, selectedFeatures)
        {
            this.numOfClasses = numOfClasses;

            this.SkipClassValueOutOfRange = true;
        }

        #endregion
        
        #region Private methods

        /// <summary>
        /// Check if the class ID is allowed
        /// </summary>
        /// <param name="classId">Class ID</param>
        /// <returns>Check result</returns>
        private bool CheckClassId(int classId)
        {
            if (classId >= numOfClasses)
            {
                if (this.SkipClassValueOutOfRange)
                {
                    return false;
                }
                else
                {
                    throw new DatasetFormatException(
                        string.Format("Class ID ({0}) is out of pre-defined range.", classId));
                }
            }

            return true;
        }

        #endregion
    }
}
