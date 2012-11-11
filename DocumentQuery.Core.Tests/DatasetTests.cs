using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using MicrosoftResearch.Infer.Maths;

namespace DocumentQuery.Core.Tests
{
    [TestFixture]
    public class DatasetTests
    {
        #region Dataset Tests

        [Test]
        [TestCase("0",          0,          TestName = "Integer0ShouldParse")]
        [TestCase("+3",         3,          TestName = "PositiveIntegerShouldParse")]
        [TestCase("123456789",  123456789,  TestName = "IntegerBiggerThanInt8ShouldParse")]
        [TestCase("-123",       -123,       TestName = "NegativeIntegerShouldParse")]
        [TestCase("1.234",      0,          TestName = "RealNumberShouldThrowException",        ExpectedException = typeof(DatasetFormatException))]
        [TestCase("1.23e7",     0,          TestName = "RealNumberWithEShouldThrowException",   ExpectedException = typeof(DatasetFormatException))]
        [TestCase("test",       0,          TestName = "CharacterStringShouldThrowException",   ExpectedException = typeof(DatasetFormatException))]
        [TestCase("0x10",       0,          TestName = "HexValueShouldThrowException",          ExpectedException = typeof(DatasetFormatException))]
        [TestCase("",           0,          TestName = "EmptyStringThrowException",             ExpectedException = typeof(DatasetFormatException))]
        public void ParseClassIdTest(string input, int expected)
        {
            Assert.That(Dataset.ParseClassId(input), Is.EqualTo(expected));
        }

        [Test]
        [TestCase("qid:WT04-170",   "WT04-170", TestName = "CorrectFormatShouldParse")]
        [TestCase("qid:12345",      "12345",    TestName = "NumericQueryIdShouldParse")]
        [TestCase("qid:WT04:170",   "WT04:170", TestName = "QueryIdWithColonShouldParse")]
        [TestCase("qid:",           "",         TestName = "EmptyQueryIdShouldThrowException",      ExpectedException = typeof(DatasetFormatException))]
        [TestCase("pid:WT04-170",   "",         TestName = "WrongPrefixShouldThrowException",       ExpectedException = typeof(DatasetFormatException))]
        [TestCase("qid=WT04-170",   "",         TestName = "InvalidFormat1ShouldThrowException",    ExpectedException = typeof(DatasetFormatException))]
        [TestCase("",               "",         TestName = "EmptyStringShouldThrowException",       ExpectedException = typeof(DatasetFormatException))]
        public void ParseQueryIdTest(string input, string expected)
        {
            Assert.That(Dataset.ParseQueryId(input), Is.EqualTo(expected));
        }

        [Test]
        [TestCase(new string[] { "#docid", "=", "G31-19-4091944" }, "G31-19-4091944",   
            TestName = "CorretFormat1ShouldParse")]
        [TestCase(new string[] { "#docid", "=", "G31=19=4091944" }, "G31=19=4091944",
            TestName = "CorretFormat2ShouldParse")]
        [TestCase(new string[] { "#docid=G31-19-4091944" },         "",
            TestName = "NoSpacesShouldThrowException",              ExpectedException = typeof(DatasetFormatException))]
        [TestCase(new string[] { "#docid", "=G31-19-4091944" },     "",
            TestName = "NoSpacesAfterEqualShouldThrowException",    ExpectedException = typeof(DatasetFormatException))]
        [TestCase(new string[] { "#docid=", "G31-19-4091944" },     "",
            TestName = "NoSpacesBeforeEqualShouldThrowException",   ExpectedException = typeof(DatasetFormatException))]
        [TestCase(new string[] { "#docid", ":", "G31-19-4091944" }, "",
            TestName = "NotEqualSignShouldThrowException",          ExpectedException = typeof(DatasetFormatException))]
        [TestCase(new string[] { "#docid", "=", "" },               "",
            TestName = "EmptyDocIdShouldThrowException",            ExpectedException = typeof(DatasetFormatException))]
        public void ParseDocumentIdTest(string[] input, string expected)
        {
            Assert.That(Dataset.ParseDocumentId(input), Is.EqualTo(expected));
        }

        [Test]
        [TestCase("33:0.732063",    33, 0.732063, TestName = "CorrectFormat1ShouldParse")]
        [TestCase("1:0.000000",     1,  0.000000, TestName = "CorrectFormat2ShouldParse")]
        [TestCase("24:0.168218",    28, 0.000000, TestName = "WrongFeatureNumberShouldThrowException",  ExpectedException = typeof(DatasetFormatException))]
        [TestCase("12:",            12, 0.000000, TestName = "EmptyFeatureValueShouldThrowException",   ExpectedException = typeof(DatasetFormatException))]
        public void ParseFeatureTest(string input, int index, double expected)
        {
            Assert.IsTrue(Math.Abs(Dataset.ParseFeature(input, index) - expected) <= 0.0000001);
        }

        [Test]
        [TestCase(new string[]{ "1:0.006077", "2:0.010417", "3:0.076923", "4:0.000000", "5:0.008108", "6:0.196020", "7:0.265382", "8:0.250778", "9:0.264915", "10:0.194877" }, 10,
            new double[]{  0.006077, 0.010417, 0.076923, 0.000000, 0.008108, 0.196020, 0.265382, 0.250778, 0.264915, 0.194877 },
            TestName = "TenInFormatFeaturesShouldParse")]
        [TestCase(new string[] { "1:0.007427", "2:0.005208", "3:0.038462", "4:0.000000", "5:0.008108", "6:0.196020", "7:0.265382", "8:0.250778" }, 11,
            new double[] {},
            TestName = "DefinedNumberOfFeaturesAreGreaterShouldThrowIndexOutOfRange",
            ExpectedException = typeof(IndexOutOfRangeException))]
        [TestCase(new string[] { "1:0.007427", "2:0.005208", "3:0.038462", "4:0.000000", "5:0.008108", "6:0.196020", "7:0.265382", "8:0.250778" }, 6,
            new double[] { 0.007427, 0.005208, 0.038462, 0.000000, 0.008108, 0.196020 },
            TestName = "DefinedNumberOfFeaturesAreSmallerShouldParse")]
        [TestCase(new string[] { "1:0.007427", "4:0.000000", "5:0.008108", "2:0.005208", "3:0.038462", "6:0.196020" }, 6,
            new double[] { },
            TestName = "FeaturesNotInOrderShouldThrowException",
            ExpectedException = typeof(DatasetFormatException))]
        [TestCase(new string[] { "a:0.007427", "b:0.005208", "c:0.038462", "4:0.000000", "5:0.008108", "6:0.196020" }, 6,
            new double[] { 0.007427, 0.005208, 0.038462, 0.000000, 0.008108, 0.196020 },
            TestName = "InvalidFeatureNumberShouldThrowException",
            ExpectedException = typeof(DatasetFormatException))]
        [TestCase(new string[] { "1:0.007abc", "2:0.005208", "3:0.038462", "4:0.000000", "5:0.008108", "6:0.196020" }, 6,
            new double[] { 0.007427, 0.005208, 0.038462, 0.000000, 0.008108, 0.196020 },
            TestName = "InvalidFeatureValueShouldThrowException",
            ExpectedException = typeof(DatasetFormatException))]
        [TestCase(new string[] { "1=0.007427", "2=0.005208", "3=0.038462", "4=0.000000", "5=0.008108", "6=0.196020" }, 6,
            new double[] { 0.007427, 0.005208, 0.038462, 0.000000, 0.008108, 0.196020 },
            TestName = "InvalidFeatureFormatShouldThrowException",
            ExpectedException = typeof(DatasetFormatException))]
        public void ParseFeaturesTest(string[] input, int numOfFeatures, double[] expected)
        {
            var dataset = new ClassifiedDataset("nosuchfile.txt", numOfFeatures, 2);

            double[] actual = dataset.ParseFeatures(input);

            VerifyParsedFeatureValues(expected, actual);
        }

        [Test]
        [TestCase(new string[] { "1:0.006077", "2:0.010417", "3:0.076923", "4:0.000000", "5:0.008108", "6:0.196020", "7:0.265382", "8:0.250778", "9:0.264915", "10:0.194877" }, 10,
            new int[] { 2, 5, 6, 8 },
            new double[] { 0.010417, 0.008108, 0.196020, 0.250778 },
            TestName = "InFormatFourSelectedFeaturesShouldParse")]
        [TestCase(new string[] { "1:0.006077", "2:0.010417", "3:0.076923", "4:0.000000", "5:0.008108", "6:0.196020", "7:0.265382", "8:0.250778", "9:0.264915", "10:0.194877" }, 10,
            new int[] { 5, 8, 2, 6 },
            new double[] { 0.008108, 0.250778, 0.010417, 0.196020 },
            TestName = "InFormatNotOrderedFourSelectedFeaturesShouldParse")]
        public void ParseSelectedFeaturesTest(string[] input, int numOfFeatures, int[] featureSelection, double[] expected)
        {
            var dataset = new ClassifiedDataset("nosuchfile.txt", numOfFeatures, featureSelection, 2);

            double[] actual = dataset.ParseSelectedFeatures(input);

            VerifyParsedFeatureValues(expected, actual);
        }

        [Test]
        [TestCase("0 qid:WT04-176 1:0.028359 2:0.005208 3:0.000000 4:0.047619 5:0.029054 6:0.075475 7:0.023407 8:0.032020 9:0.037712 10:0.075876 #docid = G22-48-2409366", 10,
            0, "WT04-176", new double[] { 0.028359, 0.005208, 0.000000, 0.047619, 0.029054, 0.075475, 0.023407, 0.032020, 0.037712, 0.075876 }, "G22-48-2409366",
            TestName = "WellFormatedRecordShouldWork")]
        [TestCase("0 qid:WT04-176 1:0.028359 2:0.005208 3:0.000000 4:0.047619 5:0.029054 6:0.075475 7:0.023407 8:0.032020 9:0.037712 10:0.075876 #docid = G22-48-2409366", 11,
            0, "", new double[] { }, "",
            TestName = "NumOfFeaturesGreaterThanActualNumOfFeaturesShouldThrow",
            ExpectedException = typeof(DatasetFormatException))]
        [TestCase("0 qid:WT04-176 1:0.028359 2:0.005208 3:0.000000 4:0.047619 5:0.029054 6:0.075475 7:0.023407 8:0.032020 9:0.037712 10:0.075876 #docid = G22-48-2409366", 8,
            0, "", new double[] { }, "",
            TestName = "NumOfFeaturesSmallerThanActualNumOfFeaturesShouldThrow",
            ExpectedException = typeof(DatasetFormatException))]
        public void GetDataVectorTest(string input, int numOfFeatures, int expectedClassId, string expectedQueryId, double[] expectedVectorValues, string expectedDocId)
        {
            var dataset = new ClassifiedDataset("nosuchfile.txt", numOfFeatures, 2);

            DataVector actual = dataset.CreateDataVector(input);

            Assert.That(actual.ClassId, Is.EqualTo(expectedClassId));
            Assert.That(actual.QueryId, Is.EqualTo(expectedQueryId));

            VerifyVectorValues(expectedVectorValues, actual.FeatureVector);

            Assert.That(actual.DocumentId, Is.EqualTo(expectedDocId));
        }

        [Test]
        [TestCase("0 qid:WT04-197 1:0.011479 2:0.000000 3:0.000000 4:0.000000 5:0.010811 6:0.220038 7:0.197803 8:0.207579 9:0.210749 10:0.220602 11:0.015582 12:0.000000 #docid = G13-62-0592350",
            12, new int[] { 1, 3, 6, 7, 10, 11 },
            0, "WT04-197", new double[] { 0.011479, 0.000000, 0.220038, 0.197803, 0.220602, 0.015582 }, "G13-62-0592350",
            TestName = "WellFormatedRecordShouldWork")]
        [TestCase("0 qid:WT04-197 1:0.011479 2:0.000000 3:0.000000 4:0.000000 5:0.010811 6:0.220038 7:0.197803 8:0.207579 9:0.210749 10:0.220602 11:0.015582 12:0.000000 #docid = G13-62-0592350",
            12, new int[] { 9, 2, 4, 5, 12, 8 },
            0, "WT04-197", new double[] { 0.210749, 0.000000, 0.000000, 0.010811, 0.000000, 0.207579 }, "G13-62-0592350",
            TestName = "ResultsShouldBeOrderedAsTheFeatureSelection")]
        [TestCase("0 qid:WT04-197 1:0.011479 2:0.000000 3:0.000000 4:0.000000 5:0.010811 6:0.220038 7:0.197803 8:0.207579 9:0.210749 10:0.220602 11:0.015582 12:0.000000 #docid = G13-62-0592350",
            13, new int[] { 1, 3, 6, 7, 10, 11 },
            0, "", new double[] { }, "",
            TestName = "NumOfFeaturesGreaterThanActualNumOfFeaturesShouldThrow",
            ExpectedException = typeof(DatasetFormatException))]
        [TestCase("0 qid:WT04-197 1:0.011479 2:0.000000 3:0.000000 4:0.000000 5:0.010811 6:0.220038 7:0.197803 8:0.207579 9:0.210749 10:0.220602 11:0.015582 12:0.000000 #docid = G13-62-0592350",
            11, new int[] { 1, 3, 6, 7, 10, 11 },
            0, "", new double[] { }, "",
            TestName = "NumOfFeaturesSmallerThanActualNumOfFeaturesShouldThrow",
            ExpectedException = typeof(DatasetFormatException))]
        public void GetDataVectorWithFeatureSelectionTest(string input, int numOfFeatures, int[] featureSelection, int expectedClassId, string expectedQueryId, double[] expectedVectorValues, string expectedDocId)
        {
            var dataset = new ClassifiedDataset("nosuchfile.txt", numOfFeatures, featureSelection, 2);

            DataVector actual = dataset.CreateDataVector(input);

            Assert.That(actual.ClassId, Is.EqualTo(expectedClassId));
            Assert.That(actual.QueryId, Is.EqualTo(expectedQueryId));

            VerifyVectorValues(expectedVectorValues, actual.FeatureVector);

            Assert.That(actual.DocumentId, Is.EqualTo(expectedDocId));
        }

        #endregion

        #region UnclassifiedDataset Tests

        [Test]
        [TestCase(@"TestData\TestData1.txt", true,   1000,  2,  TestName = "TestDataWithOneThousandRecordsShouldWork")]
        [TestCase(@"TestData\TestData2.txt", true,   12,    2,  TestName = "UnparsableRecordShouldBeSkippedByDefault")]
        [TestCase(@"TestData\TestData3.txt", true,   18,    4,  TestName = "NoLimitOnTheRangeOfClasses")]
        [TestCase(@"TestData\TestData2.txt", false,  12,    2,  TestName = "SetSkipUnparsableRecordToFalseShouldThrowException",
            ExpectedException = typeof(DatasetFormatException))]
        public void GetDataVectorsTest(string testFile, bool skipParseError, int expectedNumOfVectors, int expectedNumOfClasses)
        {
            var dataset = new UnclassifiedDataset(testFile, 64);

            if (dataset.SkipParsingErrors != skipParseError)
            {
                dataset.SkipParsingErrors = skipParseError;
            }

            IList<DataVector> vectors = dataset.GetDataVectors();

            Assert.That(vectors.Count, Is.EqualTo(expectedNumOfVectors));
            Assert.That(vectors.All(v => v.ClassId >= 0 && v.ClassId < expectedNumOfClasses && v.QueryId != string.Empty && v.FeatureVector.Count == 64 && v.DocumentId != string.Empty));
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", 64, new int[] { 1, 2, 3, 4, 5, 6, 8, 10, 22, 23, 27, 34, 35, 36, 63, 64 }, 
            2, true, 1000,  TestName = "TestDataWithOneThousandRecordsShouldWork")]
        [TestCase(@"TestData\TestData1.txt", 64, new int[] { 3, 4, 36, 63, 8, 10, 22, 64, 5, 6, 1, 2, 23, 27, 34, 35 },
            2, true, 1000,  TestName = "NotInOrderFeatureSelectionListShouldWork")]
        [TestCase(@"TestData\TestData2.txt", 64, new int[] { 1, 2, 3, 4, 5, 6, 8, 10, 22, 23, 27, 34, 35, 36, 63, 64 },
            2, true, 12,    TestName = "UnparsableRecordShouldBeSkippedByDefault")]
        [TestCase(@"TestData\TestData3.txt", 64, new int[] { 1, 2, 3, 4, 5, 6, 8, 10, 22, 23, 27, 34, 35, 36, 63, 64 },
            4, true, 18,    TestName = "NoLimitOnTheRangeOfClasses")]
        [TestCase(@"TestData\TestData2.txt", 64, new int[] { 1, 2, 3, 4, 5, 6, 8, 10, 22, 23, 27, 34, 35, 36, 63, 64 },
            2, false, 12,   TestName = "SetSkipUnparsableRecordToFalseShouldThrowException",
            ExpectedException = typeof(DatasetFormatException))]
        public void GetDataVectorsWithFeatureSelectionTest(string testFile, int numOfFeatures, int[] featureSelection, int expectedNumOfClasses, bool skipParseError, int expectedNumOfVectors)
        {
            var dataset = new UnclassifiedDataset(testFile, numOfFeatures, featureSelection);
            
            if (dataset.SkipParsingErrors != skipParseError)
            {
                dataset.SkipParsingErrors = skipParseError;
            }

            IList<DataVector> vectors = dataset.GetDataVectors();

            Assert.That(vectors.Count, Is.EqualTo(expectedNumOfVectors));
            Assert.That(vectors.All(v => v.ClassId >= 0 && v.ClassId < expectedNumOfClasses && v.QueryId != string.Empty && v.FeatureVector.Count == featureSelection.Count() && v.DocumentId != string.Empty));
        }

        #endregion

        #region ClassifiedDataset Tests

        [Test]
        [TestCase(10, new int[] { 2, 5, 6, 9 }, TestName = "CreateWithFeatureSelectionShouldPass")]
        [TestCase(10, new int[] { 6, 5, 9, 2 }, TestName = "CreateWithNotOrderedFeatureSelectionShouldPass")]
        [TestCase(10, new int[] { }, TestName = "CreateWithEmptyFeatureSelectionShouldPass")]
        [TestCase(10, new int[] { 2, 5, 6, 6 }, TestName = "DuplicatedSelectedFeaturesShouldThrowArgumentException", ExpectedException = typeof(ArgumentException))]
        [TestCase(8, new int[] { 2, 5, 6, 9 }, TestName = "SelectOutOfRangeFeaturesShouldThrowOutOfRangeException", ExpectedException = typeof(IndexOutOfRangeException))]
        public void CreationWithFeatureSelectionTest(int numOfFeatures, int[] featureSelection)
        {
            ClassifiedDataset dataset = new ClassifiedDataset("nosuchfile.txt", numOfFeatures, featureSelection, 2);
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", true, true, new int[] { 997, 3 },
            TestName = "OneThousandVectorShouldWork")]
        [TestCase(@"TestData\TestData2.txt", true, true, new int[] { 9, 3 },
            TestName = "UnparsableRecordShouldBeSkippedByDefault")]
        [TestCase(@"TestData\TestData3.txt", true, true, new int[] { 15, 1 },
            TestName = "OutOfRangeClassesShouldBeSkippedByDefault")]
        [TestCase(@"TestData\TestData2.txt", false, true, new int[] { },
            TestName = "SetSkipUnparsableRecordToFalseShouldThrowException",
            ExpectedException = typeof(DatasetFormatException))]
        [TestCase(@"TestData\TestData3.txt", true, false, new int[] { },
            TestName = "SetSkipOutOfRangeClassesToFalseShouldThrowException",
            ExpectedException = typeof(DatasetFormatException))]
        public void GetClassifiedVectorsTest(string filePath, bool skipParseError, bool skipOutOfRangeClass, int[] expectedVectorCounts)
        {
            ClassifiedDataset dataset = new ClassifiedDataset(filePath, 64, 2);

            if (dataset.SkipParsingErrors != skipParseError)
            {
                dataset.SkipParsingErrors = skipParseError;
            }

            if (dataset.SkipClassValueOutOfRange != skipOutOfRangeClass)
            {
                dataset.SkipClassValueOutOfRange = skipOutOfRangeClass;
            }

            IList<Vector>[] classifiedVectors = dataset.GetClassifiedVectors();

            Assert.That(classifiedVectors.Length, Is.EqualTo(2));

            for (int i = 0; i < 2; i++)
            {
                Assert.That(classifiedVectors[i].Count, Is.EqualTo(expectedVectorCounts[i]));
            }
        }

        [Test]
        [TestCase(@"TestData\TestData1.txt", 2, 500, 2)]
        [TestCase(@"TestData\TestData1.txt", 10, 100, 10)]
        [TestCase(@"TestData\TestData1.txt", 100, 10, 100)]
        [TestCase(@"TestData\TestData1.txt", 333, 4, 1)]
        public void GetClassifiedVectorsInChunksTest(string filePath, int chunkSize, int expectedNumOfChunk, int expectedLastChunkSize)
        {
            ClassifiedDataset dataset = new ClassifiedDataset(filePath, 64, 2);

            int actualNumOfChunk = 0;

            foreach (var chunk in dataset.GetClassifiedVectorsInChunks(chunkSize))
            {
                actualNumOfChunk++;

                Assert.That(chunk.Length, Is.EqualTo(2));

                if (actualNumOfChunk == expectedNumOfChunk)
                {
                    Assert.That(chunk.Sum(c => c.Count), Is.EqualTo(expectedLastChunkSize));
                }
                else
                {
                    Assert.That(chunk.Sum(c => c.Count), Is.EqualTo(chunkSize));
                }
            }

            Assert.That(expectedNumOfChunk, Is.EqualTo(actualNumOfChunk));
        }

        #endregion

        #region Private methods
        private void VerifyParsedFeatureValues(double[] expected, double[] actual)
        {
            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < actual.Length; i++)
            {
                Assert.IsTrue(Math.Abs(expected[i] - actual[i]) <= 0.0000001);
            }
        }

        private void VerifyVectorValues(double[] expected, Vector actual)
        {
            Assert.AreEqual(expected.Length, actual.Count);

            for (int i = 0; i < actual.Count; i++)
            {
                Assert.IsTrue(Math.Abs(expected[i] - actual[i]) <= 0.0000001);
            }
        }
        #endregion
    }
}
