version 1.0

import "https://raw.githubusercontent.com/lilab-bcb/cumulus/master/workflows/cellranger/cellranger_vdj_create_reference.wdl" as cvcr

workflow cellranger_vdj_create_reference {
    input {
        # Output directory, gs URL
        String output_directory

        # Input genome reference in either FASTA or FASTA.gz format
        File input_fasta
        # Input gene annotation file in either GTF or GTF.gz format
        File input_gtf
        # Genome reference name. New reference will be stored in a folder named genome
        String genome
        # reference version string
        String ref_version = ""

        # Which docker registry to use
        String docker_registry = "quay.io/cumulus"
        # 6.1.1, 6.0.2, 6.0.1
        String cellranger_version = "6.1.1"

        # Disk space in GB
        Int disk_space = 100
        # Google cloud zones, default to Roche Science Cloud zones
        String zones = "us-west1-a us-west1-b us-west1-c"
        # Memory string
        String memory = "32G"

        # Number of preemptible tries
        Int preemptible = 2
        # Max number of retries for AWS instance
        Int awsMaxRetries = 5
        # Backend
        String backend = "aws"
    }

    call cvcr.cellranger_vdj_create_reference as cellranger_vdj_create_reference {
        input:
            output_directory = output_directory,
            input_fasta = input_fasta,
            input_gtf = input_gtf,
            genome = genome,
            ref_version = ref_version,
            docker_registry = docker_registry,
            cellranger_version = cellranger_version,
            disk_space = disk_space,
            zones = zones,
            memory = memory,
            preemptible = preemptible,
            awsMaxRetries = awsMaxRetries,
            backend = backend
    }

    output {
        File output_reference = cellranger_vdj_create_reference.output_reference
    }
}
