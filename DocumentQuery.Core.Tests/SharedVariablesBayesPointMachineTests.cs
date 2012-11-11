using System.Linq;

using NUnit.Framework;

using DocumentQuery.Core.SharedVariablesBayesPointMachine;

namespace DocumentQuery.Core.Tests
{
    [TestFixture]
    public class SharedVariablesBayesPointMachineTests
    {
        #region TrainModel tests

        [Test]
        [TestCase(@"TestData\TestData1.txt", 100,   10, TestName = "OneThousandRecords")]
        [TestCase(@"TestData\TestData2.txt", 3,     4,  TestName = "TwelveRecords")]
        [TestCase(@"TestData\TestData2.txt", 5,     4,  TestName = "NotEnoughRecordsForChunks")]
        public void TrainModelTest(string trainFile, int chunkSize, int numOfChunk)
        {
            var model = new TrainModel(numOfChunk, 2, 64, 0.1);
            var dataset = new ClassifiedDataset(trainFile, 64, 2);

            int count = 0;

            foreach (var chunk in dataset.GetClassifiedVectorsInChunks(chunkSize))
            {
                model.Train(chunk, count);
                if (++count == numOfChunk)
                {
                    break;
                }
            }

            var weights = model.GetWeights();

            Assert.That(weights.Length, Is.EqualTo(2));

            foreach (var w in weights)
            {
                Assert.IsNotNull(w);
            }
        }

        [TestCase(@"TestData\TestData1.txt", 10, 100, 
            new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
            TestName = "TwentyFeatures")]
        [TestCase(@"TestData\TestData1.txt", 100, 10,
            new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 },
            TestName = "FourtyFeatures")]
        public void TrainModelWithSelectionTest(string trainFile, int chunkSize, int numOfChunk, int[] featureSelection)
        {
            var model = new TrainModel(numOfChunk, 2, featureSelection.Length, 0.1);
            var dataset = new ClassifiedDataset(trainFile, 64, featureSelection, 2);

            int count = 0;

            foreach (var chunk in dataset.GetClassifiedVectorsInChunks(chunkSize))
            {
                model.Train(chunk, count);
                if (++count == numOfChunk)
                {
                    break;
                }
            }

            var weights = model.GetWeights();

            Assert.That(weights.Length, Is.EqualTo(2));

            foreach (var w in weights)
            {
                Assert.IsNotNull(w);
            }
        }

        #endregion

        #region TestModel tests

        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData2.txt", 98, 10)]
        public void TrainModelAndTestModelTest(string trainFile, string testFile, int chunkSize, int numOfChunk)
        {
            var trainModel = new TrainModel(numOfChunk, 2, 64, 0.1);
            var trainDataset = new ClassifiedDataset(trainFile, 64, 2);

            var testModel = new TestModel(numOfChunk, 2, 0.1, trainModel.GetWeights());
            var testDataset = new UnclassifiedDataset(testFile, 64);

            int count = 0;

            foreach (var chunk in trainDataset.GetClassifiedVectorsInChunks(chunkSize))
            {
                trainModel.Train(chunk, count);
                if (++count == numOfChunk)
                {
                    break;
                }
            }

            var results = testModel.Test(testDataset.GetDataVectors().Select(v => v.FeatureVector).ToArray(), 0);
            
        }

        #endregion

        #region Machine tests

        [Test]
        [TestCase(@"TestData\TestData1.txt", 1, 1800, @"TestData\TestData2.txt", 12)]
        [TestCase(@"TestData\TestData1.txt", 10, 100, @"TestData\TestData2.txt", 12)]
        [TestCase(@"TestData\TestData1.txt", 100, 10, @"TestData\TestData2.txt", 12)]
        [TestCase(@"TestData\TestData1.txt", 30, 40,  @"TestData\TestData2.txt", 12)]
        public void MachineTest(string trainFile, int numOfChunk, int sizeOfChunk, string testFile, int expectedResultLength)
        {
            var machine = new Machine(2, 64, numOfChunk);

            machine.Train(trainFile, sizeOfChunk);

            var results = machine.Test(testFile);

            Assert.That(results.Length, Is.EqualTo(expectedResultLength));
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", 10, 100, @"TestData\TestData2.txt", new int[] { 1, 3, 5, 7, 9, 11, 13, 15, 55, 62 }, 12)]
        public void MachineWithFeatureSelectionTest(string trainFile, int numOfChunk, int sizeOfChunk, string testFile, int[] featureSelection, int expectedResultLength)
        {
            var machine = new Machine(2, 64, numOfChunk, featureSelection);

            machine.Train(trainFile, sizeOfChunk);

            var results = machine.Test(testFile);

            Assert.That(results.Length, Is.EqualTo(expectedResultLength));
        }
        
        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData3.txt")]
        public void TwoMachineWithSameDataShouldGetSameResultsTest(string trainFile, string testFile)
        {
            var machine1 = new Machine(2, 64, 100);
            machine1.Train(trainFile, 10);

            var machine2 = new Machine(2, 64, 100);
            machine2.Train(trainFile, 10);

            var results1 = machine1.Test(testFile);
            var results2 = machine2.Test(testFile);

            Assert.AreEqual(results1.Length, results2.Length);

            for (int i = 0; i < results1.Length; i++)
            {
                Assert.AreEqual(results1[i], results2[i]);
            }
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 3, 5, 7, 9, 11, 13, 15, 55, 62 }, @"TestData\TestData3.txt")]
        public void TwoMachineWithSameDataAndFeatureSelectionShouldGetSameResultsTest(string trainFile, int[] featureSelection, string testFile)
        {
            var machine1 = new Machine(2, 64, 100, featureSelection);
            machine1.Train(trainFile, 10);

            var machine2 = new Machine(2, 64, 100, featureSelection);
            machine2.Train(trainFile, 10);

            var results1 = machine1.Test(testFile);
            var results2 = machine2.Test(testFile);

            Assert.AreEqual(results1.Length, results2.Length);

            for (int i = 0; i < results1.Length; i++)
            {
                Assert.AreEqual(results1[i], results2[i]);
            }
        }
        #endregion
    }
}
