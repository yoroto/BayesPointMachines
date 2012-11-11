using System.Collections.Generic;
using System.IO;

namespace DocumentQuery.Core
{
    /// <summary>
    /// The dataset loader to output a vector list.
    /// </summary>
    public class UnclassifiedDataset : Dataset
    {
        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="filePath">The file path of the training data.</param>
        /// <param name="numOfFeatures">The number of features in each data record.</param>
        public UnclassifiedDataset(string filePath, int numOfFeatures)
            : base(filePath, numOfFeatures)
        { }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="filePath">The file path of the training data.</param>
        /// <param name="numOfFeatures">The number of features in each data record.</param>
        /// <param name="featureSelection">The feature selection.</param>
        public UnclassifiedDataset(string filePath, int numOfFeatures, int[] featureSelection)
            : base(filePath, numOfFeatures, featureSelection)
        { }

        #endregion

        #region Public methods

        /// <summary>
        /// Return the list of data vector from the training data file.
        /// </summary>
        /// <returns>Data vector list.</returns>
        public IList<DataVector> GetDataVectors()
        {
            IList<DataVector> list = new List<DataVector>();

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

                    list.Add(dataVector);
                }
            }

            return list;
        }

        /// <summary>
        /// Return the list of data vector from the training data file in chunks.
        /// </summary>
        /// <param name="chunkSize">The number of data vectors in each chunk.</param>
        /// <returns>A enumerator of data vector lists.</returns>
        public IEnumerable<IList<DataVector>> GetDataVectorInChunk(int chunkSize)
        {
            using (var sr = new StreamReader(this.filepath))
            {
                bool fileEnded = false;
                while (!fileEnded)
                {
                    IList<DataVector> vectors = new List<DataVector>();

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
                        
                        vectors.Add(dataVector);
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
    }
}
