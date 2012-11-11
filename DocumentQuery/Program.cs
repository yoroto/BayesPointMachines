using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using DocumentQuery.Core;

namespace DocumentQuery
{
    /// <summary>
    /// The program train three Bayes Point Machine models with the same training data,
    /// and then uses them to produce predictions of the same test data. The results
    /// are saved to a file.
    /// </summary>
    internal class Program
    {
        private const int DefaultChunkSize = 100;
        private const int DefaultNumberOfChunk = 150;
        private const int DefaultNumOfFeatures = 64;
        private const double DefaultNoise = 0.1;

        private static void Main(string[] args)
        {
            if (args[0] == "-h")
            {
                PrintHelp();
                return;
            }

            if (args.Length < 6
                || args[0] != "-t"
                || args[2] != "-p"
                || args[4] != "-r")
            {
                PrintFormatError();
                return;
            }

            string trainFile = args[1];
            string testFile = args[3];
            string resultFile = args[5];

            int chunkSize = DefaultChunkSize;
            int numOfChunk = DefaultNumberOfChunk;
            int numOfFeatures = DefaultNumOfFeatures;
            double noise = DefaultNoise;

            int[] featureSelection = new int[] {};

            int i = 6;
            while (i + 1 < args.Length)
            {
                switch (args[i])
                {
                    case "-s":
                        if (!int.TryParse(args[i + 1], out chunkSize))
                        {
                            PrintFormatError();
                            return;
                        }
                        i = i + 2;
                        break;
                    case "-c":
                        if (!int.TryParse(args[i + 1], out numOfChunk))
                        {
                            PrintFormatError();
                            return;
                        }
                        i = i + 2;
                        break;
                    case "-n":
                        if (!double.TryParse(args[i + 1], out noise))
                        {
                            PrintFormatError();
                            return;
                        }
                        i = i + 2;
                        break;
                    case "-v":
                        if (!int.TryParse(args[i + 1], out numOfFeatures))
                        {
                            PrintFormatError();
                            return;
                        }
                        i = i + 2;
                        break;
                    case "-f":
                        string[] fs = args[i + 1].Split(new char[] {':'});
                        featureSelection = new int[fs.Length];
                        for (int j = 0; j < fs.Length; j++)
                        {
                            int f;
                            if (int.TryParse(fs[j], out f))
                            {
                                featureSelection[j] = f;
                            }
                            else
                            {
                                PrintFormatError();
                                return;
                            }
                        }
                        i = i + 2;
                        break;
                    default:
                        PrintFormatError();
                        return;
                }
            }

            try
            {
                Run(trainFile, testFile, resultFile, numOfFeatures, featureSelection, noise, chunkSize, numOfChunk);
            }
            catch (Exception e)
            {
                Console.WriteLine("Error!");
                Console.WriteLine("Error info:");
                Console.WriteLine(e.Message);
            }
        }

        private static void PrintFormatError()
        {
            Console.WriteLine("Command format error!");
            PrintHelp();
        }

        private static void PrintHelp()
        {
            Console.WriteLine();
            Console.WriteLine(" Usage:");
            Console.WriteLine("    predict -t <train file> -p <test file> -r <result>");
            Console.WriteLine(" Other options:");
            Console.WriteLine("      -s <size of training chunks>");
            Console.WriteLine("      -c <number of training chunks>");
            Console.WriteLine("         This option is for shared-variable mutli-class");
            Console.WriteLine("         Bayes Point Machine only.");
            Console.WriteLine("      -f <feature selection>");
            Console.WriteLine("         Specify a list of selected features in a colon-");
            Console.WriteLine("         separated list. Example: 1:2:3:7");
        }

        private static void Run(string trainFile, string testFile, string resultFile, int numOfFeatures,
                                int[] featureSelection, double noise, int chunkSize, int numOfChunks)
        {
            Console.WriteLine("Started training the simple Bayes Point Machine...");
            var simpleBpm = new SimpleBayesPointMachine(numOfFeatures, featureSelection, noise);
            simpleBpm.Train(trainFile, chunkSize);
            Console.WriteLine("Finished training the simple Bayes Point Machine.");

            Console.WriteLine("Started training the multi-class Bayes Point Machine.");
            var bpm = new DocumentQuery.Core.MultiClassBayesPointMachine.Machine(2, numOfFeatures, featureSelection,
                                                                                 noise);
            bpm.Train(trainFile, chunkSize);
            Console.WriteLine("Finished training the multi-class Bayes Point Machine.");

            Console.WriteLine("Started training the shared-variables multi-class Bayes Point Machine.");
            var sharedBpm = new DocumentQuery.Core.SharedVariablesBayesPointMachine.Machine(2, numOfFeatures,
                                                                                            numOfChunks,
                                                                                            featureSelection, noise);
            sharedBpm.Train(trainFile, chunkSize);
            Console.WriteLine("Started training the shared-variables multi-class Bayes Point Machine.");

            using (var sw = new StreamWriter(resultFile))
            {
                var dataset = (featureSelection.Length == 0)
                    ? new UnclassifiedDataset(testFile, numOfFeatures)
                    : new UnclassifiedDataset(testFile, numOfFeatures, featureSelection);

                Console.WriteLine("Started testing...");
                foreach (var chunk in dataset.GetDataVectorInChunk(chunkSize))
                {
                    var classes = chunk.Select(v => v.ClassId).ToArray();
                    var vectors = chunk.Select(v => v.FeatureVector).ToArray();

                    var simpleBmpResults = simpleBpm.Test(vectors);
                    var bmpResults = bpm.Test(vectors);
                    var sharedBmpResults = sharedBpm.Test(vectors);

                    for (int i = 0; i < chunkSize; i++)
                    {
                        sw.WriteLine("{0} {1}\t{2}\t{3}", classes[i], simpleBmpResults[i], bmpResults[i],
                                     sharedBmpResults[i]);
                    }
                }
            }
            Console.WriteLine("Done!");
            Console.WriteLine("Results have been written to {0}", resultFile);
        }
    }
}
