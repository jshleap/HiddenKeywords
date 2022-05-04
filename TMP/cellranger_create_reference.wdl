version 1.0

import "https://raw.githubusercontent.com/lilab-bcb/cumulus/master/workflows/cellranger/cellranger_create_reference.wdl" as ccr

workflow cellranger_create_reference {
    input {
        # Output directory, gs URL
        String output_directory

        # A sample sheet in CSV format allows users to specify more than 1 genomes to build references (e.g. human and mouse). If a sample sheet is provided, input_fasta, input_gtf, and attributes will be ignored.
        File? input_sample_sheet
        # Input gene annotation file in either GTF or GTF.gz format
        String? input_gtf
        # Input genome reference in either FASTA or FASTA.gz format
        String? input_fasta
        # Genome reference name. New reference will be stored in a folder named genome
        String? genome
        # A list of key:value pairs separated by ;. If this option is not None, cellranger mkgtf will be called to filter the user-provided GTF file. See 10x filter with mkgtf for more details
        String? attributes
        # If we want to build pre-mRNA references, in which we use full length transcripts as exons in the annotation file. We follow 10x build Cell Ranger compatible pre-mRNA Reference Package to build pre-mRNA references
        Boolean pre_mrna = false
        # reference version string
        String? ref_version

        # Which docker registry to use
        String docker_registry = "quay.io/cumulus"
        # 6.1.1, 6.0.2, 6.0.1
        String cellranger_version = "6.1.1"

        # Disk space in GB
        Int disk_space = 100
        # Google cloud zones, default to Roche Science Cloud zones
        String zones = "us-west1-a us-west1-b us-west1-c"
        # Number of CPUs
        Int num_cpu = 32
        # Memory string
        String memory = "32G"

        # Number of preemptible tries
        Int preemptible = 2
        # Max number of retries for AWS instance
        Int awsMaxRetries = 5
        # Backend
        String backend = "aws"
    }

    call ccr.cellranger_create_reference as cellranger_create_reference {
        input:
            output_directory = output_directory,
            input_sample_sheet = input_sample_sheet,
            input_gtf = input_gtf,
            input_fasta = input_fasta,
            genome = genome,
            attributes = attributes,
            pre_mrna = pre_mrna,
            ref_version = ref_version,
            docker_registry = docker_registry,
            cellranger_version = cellranger_version,
            disk_space = disk_space,
            zones = zones,
            num_cpu = num_cpu,
            memory = memory,
            preemptible = preemptible,
            awsMaxRetries = awsMaxRetries,
            backend = backend
    }

    output {
        File output_reference = cellranger_create_reference.output_reference
    }
}
