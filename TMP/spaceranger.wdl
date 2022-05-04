version 1.0

import "https://raw.githubusercontent.com/lilab-bcb/cumulus/master/workflows/spaceranger/spaceranger_workflow.wdl" as srw

workflow spaceranger_workflow {
    input {
        # 5 - 14 columns (Sample, Reference, Flowcell, Lane, Index, [Image, DarkImage, ColorizedImage, Slide, Area, SlideFile, ReorientImages, LoupeAlignment, TargetPanel])
        File input_csv_file
        # Output directory, gs URL
        String output_directory

        # If run mkfastq
        Boolean run_mkfastq = true
        # If run count
        Boolean run_count = true

        # for mkfastq

        # Whether to delete input_bcl_directory, default: false
        Boolean delete_input_bcl_directory = false
        # Number of allowed mismatches per index
        Int? mkfastq_barcode_mismatches

        # Referece index TSV
        File acronym_file = "s3://gred-cumulus-ref/resources/cellranger/index.tsv"

        # For spaceranger count

        # If generate bam outputs. This is also a spaceranger argument.
        Boolean no_bam = false
        # Perform secondary analysis of the gene-barcode matrix (dimensionality reduction, clustering and visualization). Default: false. This is also a spaceranger argument.
        Boolean secondary = false

        # Space Ranger version: 1.3.0
        String spaceranger_version = "1.3.0"
        # Config version: 0.2
        String config_version = "0.2"

        # Which docker registry to use: quay.io/cumulus (default) or cumulusprod
        String docker_registry = "quay.io/cumulus"
        # spaceranger mkfastq registry, default to gcr.io/broad-cumulus
        String mkfastq_docker_registry = "752311211819.dkr.ecr.us-west-2.amazonaws.com"
        # Google cloud zones, default to Roche Science Cloud zones
        String zones = "us-west1-a us-west1-b us-west1-c"
        # Number of cpus per spaceranger and spaceranger job
        Int num_cpu = 32
        # Memory string
        String memory = "120G"

        # Optional disk space for mkfastq.
        Int mkfastq_disk_space = 1500
        # Optional disk space needed for count.
        Int count_disk_space = 500

        # Number of preemptible tries
        Int preemptible = 2
        # Number of maximum retries when running on AWS
        Int awsMaxRetries = 5
        # Backend
        String backend = "aws"
    }

    call srw.spaceranger_workflow as spaceranger_workflow {
        input:
            input_csv_file = input_csv_file,
            output_directory = output_directory,
            run_mkfastq = run_mkfastq,
            run_count = run_count,
            delete_input_bcl_directory = delete_input_bcl_directory,
            mkfastq_barcode_mismatches = mkfastq_barcode_mismatches,
            acronym_file = acronym_file,
            no_bam = no_bam,
            secondary = secondary,
            spaceranger_version = spaceranger_version,
            config_version = config_version,
            docker_registry = docker_registry,
            mkfastq_docker_registry = mkfastq_docker_registry,
            zones = zones,
            num_cpu = num_cpu,
            memory = memory,
            mkfastq_disk_space = mkfastq_disk_space,
            count_disk_space = count_disk_space,
            preemptible = preemptible,
            awsMaxRetries = awsMaxRetries,
            backend = backend
    }
}
