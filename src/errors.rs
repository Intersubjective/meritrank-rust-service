use meritrank::MeritRankError;

#[allow(dead_code)]
// Define a new error type for better error handling
#[derive(Debug, thiserror::Error)]
pub enum GraphManipulationError {
    /// Error when failing to create an edge in the graph
    #[error("Failed to create edge: {0}")]
    EdgeCreationFailure(String),

    /// Error when failing to create a node in the graph
    #[error("Failed to create node: {0}")]
    NodeCreationFailure(String),

    /// Error when failing to create a table in the database
    #[error("Failed to create table: {0}")]
    TableCreationFailure(String),

    /// Error when failing to initiate a transaction in the database
    #[error("Failed to initiate transaction: {0}")]
    TransactionInitiationFailure(String),

    /// Error when failing to prepare a statement for the database
    #[error("Failed to prepare statement: {0}")]
    StatementPreparationFailure(String),

    /// Error when failing to commit a transaction in the database
    #[error("Failed to commit transaction: {0}")]
    TransactionCommitFailure(String),

    /// Error when a specific node could not be found in the graph
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Error when failing to extract data from the database
    #[error("Failed to extract data: {0}")]
    DataExtractionFailure(String),

    /// Error when failing to extract weight data from the database
    #[error("Failed to extract weight: {0}")]
    WeightExtractionFailure(String),

    /// Error when failing to extract record data from the database
    #[error("Failed to extract records: {0}")]
    RecordsExtractionFailure(String),

    /// Error when failing to fetch record data from the database
    #[error("Failed to fetch records: {0}")]
    FetchRecordsFailure(String),

    /// Error when failing to generate a graph from the data
    #[error("Failed to generate graph: {0}")]
    GraphGenerationFailure(String),

    /// Error when failing to write a graph to the database
    #[error("Failed to write graph: {0}")]
    GraphWriteFailure(String),

    /// Error when failing to read a graph from the database
    #[error("Failed to read graph: {0}")]
    GraphReadFailure(String),

    /// Error when a specific node name could not be found in the graph
    #[error("Node name not found: {0}")]
    NodeNameNotFound(String),

    /// Error when failing to select a node from the graph
    #[error("Failed to select node: {0}")]
    NodeSelectionFailure(String),

    /// Error when SPI operation fails. This is a transparent error, carrying the original SPI error.
    //#[error(transparent)]
    //SpiFailure(#[from] pgrx::spi::Error),

    /// Error when merit rank operation fails. This is a transparent error, carrying the original MeritRankError.
    #[error(transparent)]
    MeritRankFailure(#[from] MeritRankError),

    /// Error when failing to lock a mutex for concurrent operations
    #[error("Failed to lock mutex: {0}")]
    MutexLockFailure(String),

    #[error("Context {0} not found")]
    UnknownContextFailure(String),
}
