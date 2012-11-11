using System;
using System.Collections.Generic;

namespace DocumentQuery.Core
{
    internal static class StaticUtils
    {
        /// <summary>
        /// Extension method to allow initialize items in an array.
        /// </summary>
        /// <typeparam name="T">Item type.</typeparam>
        /// <param name="array">The array to work on.</param>
        /// <param name="provider">Constructor method.</param>
        /// <returns>An array with all items initialized with <i>provider</i>.</returns>
        public static T[] Populate<T>(this T[] array, Func<T> provider)
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = provider();
            }
            return array;
        }

        /// <summary>
        /// Extension method to create a inverted index from an array.
        /// </summary>
        /// <typeparam name="T">Item type.</typeparam>
        /// <param name="array">The array to work on.</param>
        /// <returns>The inverted index.</returns>
        public static Dictionary<T,int> ToIndexDictionary<T>(this T[] array)
        {
            var dictionary = new Dictionary<T, int>();

            for (int i = 0; i < array.Length; i++)
            {
                dictionary.Add(array[i], i);
            }

            return dictionary;
        }
    }
}
