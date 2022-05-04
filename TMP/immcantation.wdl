# WDL workflow for Immcantation changeo

version 1.0

# Main

workflow immcantation {

    input {
        # Library identifier: Path to folder with annotation and sequences
        Array[String] vdj_folders
        # Species name. One of human, mouse, rabbit, rat, or rhesus_monkey
        Array[String] species
        # Which version of the immcantation
        String version
        # Cloud backend: aws, gcp
        String backend
        # Number of preemptible tries
        Int preemptible = 2
        # Max number of retries for AWS instance
        Int awsMaxRetries = 5
    }

    String zones = "us-west1-a us-west1-b us-west1-c"
    String cumulus_private_repository = if backend == "aws" then "752311211819.dkr.ecr.us-west-2.amazonaws.com" else "gcr.io/gred-cumulus-sb-01-991a49c4"

    scatter (folders_n_species in zip(vdj_folders, species)) {
        String vdj_folder = folders_n_species.left
        String organism = folders_n_species.right
        # Run basic changeo pipeline
        call run_changeo_10x {
            input:
                vdj_folder = vdj_folder,
                organism = organism,
                backend = backend,
                version = version,
                docker_registry = cumulus_private_repository,
                zones = zones,
                preemptible = preemptible,
                awsMaxRetries = awsMaxRetries
        }
    }
}

# Basic changeo-10X pipeline
task run_changeo_10x {
    input {
        # library name
        String vdj_folder
        # species name
        String organism
        # Backed to use: aws or gcp
        String backend
        # Zones for the backend
        String zones
        # Number of preemptible tries
        Int preemptible
        # Which docker registry to use: quay.io/cumulus (default) or cumulusprod
        String docker_registry
        # Which version of the immcantation
        String version = '4.3.0'
        # Max number of retries for AWS instance
        Int awsMaxRetries
    }
    File contigs = "~{vdj_folder}/filtered_contig.fasta"
    File annotations = "~{vdj_folder}/filtered_contig_annotations.csv"
    String name = basename(vdj_folder)
    command <<<
        monitor_script.sh > monitoring.log &

        loci=$(
          awk -F',' -vcol=chain '(NR==1){colnum=-1;for(i=1;i<=NF;i++)if($(i)==col)colnum=i;}{print $(colnum)}' ~{annotations}| \
          grep -iv Multi| tail -n +2| cut -c1,2 | sort -urn | tr '[:upper:]' '[:lower:]'
        )

        changeo-10x \
        -s ~{contigs} \
        -a ~{annotations} \
        -g ~{organism} \
        -t ${loci} \
        -n ~{name} \
        -o changeo-10x_outs \
        -f airr \
        -i \
        -p 1

        # copy only files
        strato cp --backend ~{backend} changeo-10x_outs/~{name}_igblast.fmt7 ~{vdj_folder}/changeo-10x_outs/
        strato cp --backend ~{backend} changeo-10x_outs/~{name}_db-pass.tsv ~{vdj_folder}/changeo-10x_outs/
        strato cp --backend ~{backend} changeo-10x_outs/~{name}_*_productive-T.tsv ~{vdj_folder}/changeo-10x_outs/
        strato cp --backend ~{backend} changeo-10x_outs/logs/pipeline-10x.err ~{vdj_folder}/changeo-10x_outs/~{name}_changeo_10x.err
    >>>


    runtime {
        docker: "~{docker_registry}/immcantation:~{version}"
        zones: zones
        preemptible: preemptible
        maxRetries: if backend == "aws" then awsMaxRetries else 0
        cpu: 1
        memory: "4G"
    }


    output {
        # Outputs directory
        String output_changeo_directory = "~{vdj_folder}/changeo-10x_outs"
        # Monitoring file
        File monitoringLog = "monitoring.log"
    }
}