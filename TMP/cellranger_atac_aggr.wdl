version 1.0

import "https://raw.githubusercontent.com/lilab-bcb/cumulus/master/workflows/cellranger/cellranger_atac_aggr.wdl" as caa

workflow cellranger_atac_aggr {
    input {
        # Aggregate ID
        String aggr_id
        # A comma-separated list of input atac count result directories (gs urls), note that each directory should contain fragments.tsv.gz and singlecell.csv
        String input_counts_directories
        # CellRanger-atac output directory, gs url
        String output_directory
        # Index TSV file
        File acronym_file

        # Keywords or a URL to a tar.gz file
        String genome

        # Sample normalization MODE: none (default), depth, signal
        String normalize = "none"
        # Perform secondary analysis (dimensionality reduction, clustering and visualization). Default: false
        Boolean secondary = false
        # Chose the algorithm for dimensionality reduction prior to clustering and tsne: 'lsa' (default), 'plsa', or 'pca'.
        String dim_reduce = "lsa"
        # A BED file to override peak caller
        File? peaks

        # 2.0.0, 1.2.0, 1.1.0
        String cellranger_atac_version = "2.0.0"
        # Which docker registry to use: cumulusprod (default) or quay.io/cumulus
        String docker_registry = "quay.io/cumulus"

        # Google cloud zones, default to Roche Science Cloud zones
        String zones = "us-west1-a us-west1-b us-west1-c"
        # Number of cpus per cellranger job
        Int num_cpu = 64
        # Memory string, e.g. 57.6G
        String memory = "57.6G"
        # Disk space in GB
        Int disk_space = 500

        # Number of preemptible tries
        Int preemptible = 2
        # Max number of retries for AWS instance
        Int awsMaxRetries = 5
        # Backend
        String backend = "aws"
    }

    call caa.cellranger_atac_aggr as cellranger_atac_aggr {
        input:
            aggr_id = aggr_id,
            input_counts_directories = input_counts_directories,
            output_directory = output_directory,
            acronym_file = acronym_file,
            genome = genome,
            normalize = normalize,
            secondary = secondary,
            dim_reduce = dim_reduce,
            peaks = peaks,
            cellranger_atac_version = cellranger_atac_version,
            docker_registry = docker_registry,
            zones = zones,
            num_cpu = num_cpu,
            memory = memory,
            disk_space = disk_space,
            preemptible = preemptible,
            awsMaxRetries = awsMaxRetries,
            backend = backend
    }

    output {
        String output_aggr_directory = cellranger_atac_aggr.output_aggr_directory
        File output_metrics_summary = cellranger_atac_aggr.output_metrics_summary
        File output_web_summary = cellranger_atac_aggr.output_web_summary
    }
}
