version 1.0

task write_to_DataSetDB_multiome {
    input {
        Array[Array[String]?] count_outputs
        String output_directory
        File gmty_index_file
        String genome
        String project_id
        String project_authors
        String project_title
        String project_description
        File map_file
        String docker_registry
        String backend
        String zones
        Int num_cpu
        String memory
        Int preemptible
        Int disk_space
        Int awsMaxRetries
        String awsQueueArn
    }

    String output_directory_stripped = sub(output_directory, "[/\\s]+$", "")

    Boolean is_genome_uri = if sub(genome, "^(gs|s3)://.+$", "URI") == "URI" then true else false

    Map[String, String] gmty_map = read_map(gmty_index_file)
    String genomitory_id = if is_genome_uri then "" else gmty_map[genome]

    command <<<
        set -e
        export TMPDIR=/tmp

        export PROJECT_ID="~{project_id}"
        export PROJECT_AUTHORS="~{project_authors}"
        export PROJECT_TITLE="~{project_title}"
        export PROJECT_DESCRIPTION="~{project_description}"

        if [ "~{genomitory_id}" != "" ]; then
            export PROJECT_ANNOTATION="~{genomitory_id}"
        fi

        python <<CODE

        import os
        from pathlib import Path
        from subprocess import check_call
        import pandas as pd

        modality_dict = {
           "rna" : "gex",
           "crispr" : "crispr",
           "hashing" : "hashing",
           "cmo" : "cmo",
           "citeseq" : "citeseq",
           "adt" : "adt"
        }

        if "~{project_authors}" == "":
            raise Exception("'project_authors' must not be empty!")

        def mkdir_dsdb(path):
            Path(path).mkdir(parents=True, exist_ok=True)

        def check(call_args):
            print(' '.join(call_args))
            check_call(call_args)

        top_fldr = "/files"
        map_df = pd.read_csv("~{map_file}", header=0)
        for i in range(len(map_df)):
            library = map_df.loc[i, "Library"]
            sample = map_df.loc[i, "Sample"]
            datatype = map_df.loc[i, "DataType"]

            if datatype == "adt":
                dsdb_fldr1 = os.path.join(top_fldr, sample, "adt", modality_dict["hashing"])
                mkdir_dsdb(dsdb_fldr1)
                dsdb_fldr2 = os.path.join(top_fldr, sample, "adt", modality_dict["citeseq"])
                mkdir_dsdb(dsdb_fldr2)
            else:
                dsdb_fldr = os.path.join(top_fldr, sample, modality_dict[datatype])
                mkdir_dsdb(dsdb_fldr)

            if datatype == "rna":
                call_args = ["strato", "cp", "--backend", "~{backend}", "~{output_directory_stripped}/" + library + "/filtered_feature_bc_matrix.h5", dsdb_fldr + "/" + library + "_filtered_feature_bc_matrix.h5"]
                check(call_args)
            elif datatype == "adt":
                call_args = ["strato", "cp", "--backend", "~{backend}", "~{output_directory_stripped}/" + library + "/" + library + ".hashing.csv", dsdb_fldr1 + "/"]
                check(call_args)
                call_args = ["strato", "cp", "--backend", "~{backend}", "~{output_directory_stripped}/" + library + "/" + library + ".citeseq.csv", dsdb_fldr2 + "/"]
                check(call_args)
            else:
                call_args = ["strato", "cp", "--backend", "~{backend}", "~{output_directory_stripped}/" + library + "/" + library + "." + datatype + ".csv", dsdb_fldr + "/"]
                check(call_args)

        CODE

        /software/write2dsdb.sh ~{backend} /software/save_multiome.R
    >>>

    output {
        File output_dsdb_id_txt = "output.txt"
        String output_dsdb_id = read_string("output.txt")
    }

    runtime {
        docker: "~{docker_registry}/cumulus2dsdb:master"
        cpu: num_cpu
        memory: memory
        disks: "local-disk ~{disk_space} HDD"
        zones: zones
        preemptible: preemptible
        maxRetries: if backend == "aws" then awsMaxRetries else 0
        queueArn: awsQueueArn
    }
}

task write_to_DataSetDB_demux{
    input {
        Array[File] input_zarr_files
        File? index_remap_csv
        String dsdb_id
        String dsdb_project_authors
        String backend
        String docker_registry
        String zones
        Int num_cpu
        String memory
        Int disk_space
        Int preemptible
        Int awsMaxRetries
    }

    Boolean run_index_remap = if defined(index_remap_csv) then true else false

    command <<<
        set -e
        export TMPDIR=/tmp

        export DSDB_ID="~{dsdb_id}"
        if [ "~{dsdb_project_authors}" != "" ]; then
            export PROJECT_AUTHORS="~{dsdb_project_authors}"
        fi

        if [ "~{run_index_remap}" = "true" ]; then
            python /software/gen_demux_summary.py ~{sep=',' input_zarr_files} ~{index_remap_csv}
        else
            python /software/gen_demux_summary.py ~{sep=',' input_zarr_files}
        fi

        /software/write2dsdb.sh ~{backend} /software/save_demux.R
    >>>

    output {
        File dsdb_project_version_txt = "output.txt"
        String dsdb_project_version = read_string("output.txt")
    }

    runtime {
        docker: "~{docker_registry}/cumulus2dsdb:master"
        cpu: num_cpu
        memory: memory
        disks: "local-disk ~{disk_space} HDD"
        zones: zones
        preemptible: preemptible
        maxRetries: if backend == "aws" then awsMaxRetries else 0
    }
}
