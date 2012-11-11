using System.Linq;
using NUnit.Framework;

using DocumentQuery.Core.MultiClassBayesPointMachine;

namespace DocumentQuery.Core.Tests
{
    [TestFixture]
    public class MultiClassBayesPointMachineTests
    {
        #region TrainModel tests

        [Test]
        [TestCase(@"TestData\TestData1.txt", TestName = "OneThousandRecords")]
        [TestCase(@"TestData\TestData2.txt", TestName = "TwelveRecords")]
        public void TrainModelTest(string trainFile)
        {
            var model = new TrainModel(2, 0.1);
            var dataset = new ClassifiedDataset(trainFile, 64, 2);

            model.Train(dataset.GetClassifiedVectors());

            var posteriors = model.GetInferredPosterier();

            Assert.That(posteriors.Length, Is.EqualTo(2));

            foreach (var p in posteriors)
            {
                Assert.That(p.Dimension, Is.EqualTo(64));
            }
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
            TestName = "TenFeatures")]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
            TestName = "TwentyFeatures")]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 },
            TestName = "ThirtyFeatures")]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 },
            TestName = "FourtyFeatures")]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50 },
            TestName = "FiftyFeatures")]
        public void TrainModelWithFeatureSelectionTest(string trainFile, int[] featureSelection)
        {
            var model = new TrainModel(2, 0.1);
            var dataset = new ClassifiedDataset(trainFile, 64, featureSelection, 2);

            model.Train(dataset.GetClassifiedVectors());

            var posteriors = model.GetInferredPosterier();

            Assert.That(posteriors.Length, Is.EqualTo(2));

            foreach (var p in posteriors)
            {
                Assert.That(p.Dimension, Is.EqualTo(featureSelection.Length));
            }
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData4.txt")]
        public void TrainModelIncrementalTrainTest(string trainFile1, string trainFile2)
        {
            var model = new TrainModel(2, 0.1);

            var dataset1 = new ClassifiedDataset(trainFile1, 64, 2);
            model.TrainIncremental(dataset1.GetClassifiedVectors());
            var posteriors1 = model.GetInferredPosterier();

            var dataset2 = new ClassifiedDataset(trainFile2, 64, 2);
            model.TrainIncremental(dataset2.GetClassifiedVectors());
            var posteriors2 = model.GetInferredPosterier();

            Assert.AreEqual(posteriors1.Length, posteriors2.Length);

            Assert.That(posteriors1.Intersect(posteriors2).Count() < posteriors1.Length);
        }

        #endregion

        #region TestModel tests

        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData2.txt", TestName = "OneThousandRecordTrainingData")]
        [TestCase(@"TestData\TestData3.txt", @"TestData\TestData2.txt", TestName = "SmallTrainingData")]
        public void TrainModelAndTestModelTest(string trainFile, string testFile)
        {
            var trainModel = new TrainModel(2, 0.1);
            var trainDataset = new ClassifiedDataset(trainFile, 64, 2);
            trainModel.Train(trainDataset.GetClassifiedVectors());

            var testModel = new TestModel(2, 0.1);
            var testDataset = new UnclassifiedDataset(testFile, 64);
            var testData = testDataset.GetDataVectors().Select(v => v.FeatureVector).ToArray();
            var results = testModel.Test(trainModel.GetInferredPosterier(), testData);

            Assert.That(results.Length, Is.EqualTo(testData.Length));
        }

        #endregion

        #region Machine tests

        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData2.txt", 12)]
        public void MachineSingleBatchTest(string trainFile, string testFile, int expectedResultLength)
        {
            var machine = new Machine(2, 64);

            machine.Train(trainFile);

            var results = machine.Test(testFile);

            Assert.That(results.Length, Is.EqualTo(expectedResultLength));
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData2.txt", new int[] { 1, 3, 5, 7, 9, 11, 13, 15, 55, 62 }, 12)]
        public void MachineSingleBatchWithFeatureSelectionTest(string trainFile, string testFile, int[] featureSelection, int expectedResultLength)
        {
            var machine = new Machine(2, 64, featureSelection);

            machine.Train(trainFile);

            var results = machine.Test(testFile);

            Assert.That(results.Length, Is.EqualTo(expectedResultLength));
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData4.txt", @"TestData\TestData3.txt", 18)]
        public void MachineTwoBatchsTest(string trainFile1, string trainFile2, string testFile, int expectedResultLength)
        {
            var machine = new Machine(2, 64);

            machine.Train(trainFile1);
            machine.Train(trainFile2);

            var results = machine.Test(testFile);

            Assert.That(results.Length, Is.EqualTo(expectedResultLength));
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData1.txt", @"TestData\TestData3.txt")]
        public void TwoMachineWithSameDataShouldGetSameResultsTest(string trainFile1, string trainFile2, string testFile)
        {
            var machine1 = new Machine(2, 64);
            machine1.Train(trainFile1);

            var machine2 = new Machine(2, 64);
            machine2.Train(trainFile2);

            var results1 = machine1.Test(testFile);
            var results2 = machine2.Test(testFile);

            Assert.AreEqual(results1.Length, results2.Length);

            for (int i = 0; i < results1.Length; i++)
            {
                Assert.AreEqual(results1[i], results2[i]);
            }
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData1.txt", new int[] { 1, 3, 5, 7, 9, 11, 13, 15, 55, 62 }, @"TestData\TestData3.txt")]
        public void TwoMachineWithSameDataAndFeatureSelectionShouldGetSameResultsTest(string trainFile1, string trainFile2, int[] featureSelection, string testFile)
        {
            var machine1 = new Machine(2, 64, featureSelection);
            machine1.Train(trainFile1);

            var machine2 = new Machine(2, 64, featureSelection);
            machine2.Train(trainFile2);

            var results1 = machine1.Test(testFile);
            var results2 = machine2.Test(testFile);

            Assert.AreEqual(results1.Length, results2.Length);

            for (int i = 0; i < results1.Length; i++)
            {
                Assert.AreEqual(results1[i], results2[i]);
            }
        }

        [Test]
        [TestCase(@"TestData\TestData4.txt", @"TestData\TestData2.txt", 10, 12)]
        public void TrainInChunksTest(string trainFile, string testFile, int chunkSize, int expectedResultLength)
        {
            var machine = new Machine(2, 64);

            machine.Train(trainFile, chunkSize);

            var results = machine.Test(testFile);

            Assert.That(results.Length, Is.EqualTo(expectedResultLength));
        }

        [Test]
        [TestCase(@"TestData\TestData4.txt", @"TestData\TestData2.txt", new int[] { 1, 3, 5, 7, 9, 11, 13, 15, 55, 62 }, 10, 12)]
        public void TrainInChunksWithFeatureSelectionTest(string trainFile, string testFile, int[] featureSelection, int chunkSize, int expectedResultLength)
        {
            var machine = new Machine(2, 64, featureSelection);

            machine.Train(trainFile, chunkSize);

            var results = machine.Test(testFile);

            Assert.That(results.Length, Is.EqualTo(expectedResultLength));
        }

        #endregion
    }
}
