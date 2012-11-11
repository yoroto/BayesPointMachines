using System;
using System.Runtime.Serialization;

namespace DocumentQuery.Core
{
    /// <summary>
    /// A cusotmized exception for data record formatting errors.
    /// </summary>
    [Serializable]
    public class DatasetFormatException : Exception
    {
        public DatasetFormatException()
            : base() { }
    
        public DatasetFormatException(string message)
            : base(message) { }

        public DatasetFormatException(string message, Exception innerException)
            : base(message, innerException) { }

        protected DatasetFormatException(SerializationInfo info, StreamingContext context)
            : base(info, context) { }
    }
}
