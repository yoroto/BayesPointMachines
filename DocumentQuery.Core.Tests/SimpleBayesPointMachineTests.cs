using NUnit.Framework;
using MicrosoftResearch.Infer.Distributions;

namespace DocumentQuery.Core.Tests
{
    [TestFixture]
    public class SimpleBayesPointMachineTests
    {
        [Test]
        [TestCase(@"TestData\TestData1.txt", TestName = "OneThousandRecords")]
        [TestCase(@"TestData\TestData2.txt", TestName = "TwelveRecords")]
        public void TrainTest(string trainFile)
        {
            var machine = new SimpleBayesPointMachine(64);
   
            machine.Train(trainFile);
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", 100, TestName = "OneThousandRecords")]
        [TestCase(@"TestData\TestData2.txt", 3, TestName = "TwelveRecords")]
        public void TrainInChunksTest(string trainFile, int chunkSize)
        {
            var machine = new SimpleBayesPointMachine(64);

            machine.Train(trainFile, chunkSize);
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
            TestName = "TenFeatures")]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
            TestName = "TwentyFeatures")]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28 ,29, 30 },
            TestName = "ThirtyFeatures")]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 },
            TestName = "FourtyFeatures")]
        [TestCase(@"TestData\TestData1.txt", new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50 },
            TestName = "FiftyFeatures")]
        public void TrainWithFeatureSelectionTest(string trainFile, int[] featureSelection)
        {
            var machine = new SimpleBayesPointMachine(64, featureSelection);

            machine.Train(trainFile);
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", @"TestData\TestData2.txt", 12)]
        public void TrainAndTestTest(string trainFile, string testFile, int expectedNumOfResults)
        {
            var machine = new SimpleBayesPointMachine(64);

            machine.Train(trainFile);

            Bernoulli[] predictions =  machine.Test(testFile);

            Assert.That(predictions.Length, Is.EqualTo(expectedNumOfResults));
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", 50, @"TestData\TestData2.txt", 12)]
        public void TrainInChunksAndTestTest(string trainFile, int chunkSize, string testFile, int expectedNumOfResults)
        {
            var machine = new SimpleBayesPointMachine(64);

            machine.Train(trainFile, chunkSize);

            Bernoulli[] predictions = machine.Test(testFile);

            Assert.That(predictions.Length, Is.EqualTo(expectedNumOfResults));
        }
    }
}
