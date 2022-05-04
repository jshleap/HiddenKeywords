version 1.0

import "https://raw.githubusercontent.com/lilab-bcb/cumulus/master/workflows/cellranger/cellranger_atac_create_reference.wdl" as cacr

workflow cellranger_atac_create_reference {
    input {
        # Which docker registry to use
        String docker_registry = "quay.io/cumulus"
        # cellranger-atac version: 2.0.0, 1.2.0, 1.1.0
        String cellranger_atac_version = "2.0.0"

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

        # Organism name
        String organism = ""
        # Genome name
        String genome
        # GSURL for input fasta file
        File input_fasta
        # GSURL for input GTF file
        File input_gtf
        # A comma separated list of names of contigs that are not in nucleus
        String non_nuclear_contigs = "chrM"
        # Optional file containing transcription factor motifs in JASPAR format
        File? input_motifs

        # Output directory, gs URL
        String output_directory
    }

    call cacr.cellranger_atac_create_reference as cellranger_atac_create_reference {
        input:
            docker_registry = docker_registry,
            cellranger_atac_version = cellranger_atac_version,
            disk_space = disk_space,
            zones = zones,
            memory = memory,
            preemptible = preemptible,
            awsMaxRetries = awsMaxRetries,
            backend = backend,
            organism = organism,
            genome = genome,
            input_fasta = input_fasta,
            input_gtf = input_gtf,
            non_nuclear_contigs = non_nuclear_contigs,
            input_motifs = input_motifs,
            output_directory = output_directory
    }

    output {
        File output_reference = cellranger_atac_create_reference.output_reference
    }
}
