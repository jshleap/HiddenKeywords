version 1.0

workflow archr_create_arrow {
    input {
        #Archr version
        String archr_version = "1.0.1"
        # Input Fragments file (ATAC-Seq)           
        File input_fragments
        # Sample name
        String sample_name
        # Genome
        String genome
        # Number of threads
        Int num_cpu = 8
        # Minimum transcription start site (TSS) enrichment score required for a cell to pass filtering for use in downstream analyses.
        String minTSS = "4"
        # Minimum number of mapped ATAC-seq fragments required per cell to pass filtering for use in downstream analyses.
        String minFrags = "1000"
        # Maximum number of mapped ATAC-seq fragments required per cell to pass filtering for use in downstream analyses.
        String maxFrags = "Inf"
        # Output directory
        String output_directory

        # Backend
        String backend = "aws"
        # Memory string for archr count
        String memory = "32G"
        # Optional disk space needed for archr
        Int disk_space = 50
        # Number of preemptible tries
        Int preemptible = 2
        # Max number of retries for AWS instance
        Int awsMaxRetries = 5
    }

    # Google cloud zones, default to Roche Science Cloud zones
    String zones = "us-west1-a us-west1-b us-west1-c"
    String docker_registry = if backend == "aws" then "752311211819.dkr.ecr.us-west-2.amazonaws.com" else "gcr.io/gred-cumulus-sb-01-991a49c4"



    call generate_arrow_file {
       input:
           input_fragments = input_fragments,
           sample_name = sample_name,
           genome = genome,
           num_cpu = num_cpu,
           minTSS = minTSS,
           minFrags = minFrags,
           maxFrags = maxFrags,
           output_directory = output_directory,
           backend = backend,
           docker_registry = docker_registry,
           memory = memory,
           disk_space = disk_space,
           preemptible = preemptible,
           awsMaxRetries = awsMaxRetries,
           version = archr_version,
           zones = zones
    }

    output {
        File arrow_file = generate_arrow_file.arrow_file
        String qc = generate_arrow_file.qc
        File monitoringLog = generate_arrow_file.monitoringLog
    }
}

task generate_arrow_file {
    input {
        File input_fragments
        String sample_name
        String genome
        Int num_cpu
        String minTSS
        String minFrags
        String maxFrags
        String output_directory
        String backend
        String docker_registry
        String memory
        Int disk_space
        Int preemptible       
        Int awsMaxRetries
        String version
        String zones
    }

    command <<<
        set -e
        export TMPDIR=/tmp
        monitor_script.sh > monitoring.log &

        mkdir results
        create_arrow.R ~{input_fragments} ~{sample_name} ~{genome} ~{num_cpu} ~{minTSS} ~{minFrags} ~{maxFrags} results

        strato cp --backend ~{backend} results/~{sample_name}.arrow ~{output_directory}/
        strato cp -r --backend ~{backend} results/QualityControl/~{sample_name} ~{output_directory}/

    >>>

    output {
        File arrow_file = "results/~{sample_name}.arrow"
        String qc = "~{output_directory}/QualityControl/~{sample_name}"
        File monitoringLog = "monitoring.log"
    }

    runtime {
        docker: "~{docker_registry}/archr:~{version}"
        zones: zones
        preemptible: preemptible
        maxRetries: if backend == "aws" then awsMaxRetries else 0
        cpu: num_cpu
        memory: memory
        disks: "local-disk ~{disk_space} HDD"
    }
}