version 1.0

import "https://raw.githubusercontent.com/lilab-bcb/cumulus/master/workflows/demultiplexing/demultiplexing.wdl" as dmp
import "cumulus2dsdb.wdl" as dsdb

workflow demultiplexing {
    input {
        # Input CSV file describing metadata of RNA and hashtag/genetic data pairing.
        File input_sample_sheet
        # This is the output directory (gs url + path) for all results. There will be one folder per RNA-hashtag/genetic data pair under this directory.
        String output_directory
        # Reference genome name
        String genome = ""
        # demultiplexing algorithm to use for genetic-pooling data, choosing from 'souporcell' or 'popscle' (demuxlet/freemuxlet)
        String demultiplexing_algorithm = "souporcell"
        # Only demultiplex cells/nuclei with at least <min_num_genes> expressed genes
        Int min_num_genes = 100
        # Antibody index mapping file: index, sample
        File antibody_index_mapping_csv
        Boolean write_to_dsdb = true
        String? dsdb_id
        String dsdb_project_authors = ""

        # Which docker registry to use: quay.io/cumulus (default) or cumulusprod
        String docker_registry = "quay.io/cumulus"
        # Number of preemptible tries
        Int preemptible = 2
        # Number of maximum retries when running on AWS
        Int awsMaxRetries = 5
        # Backend
        String backend = "aws"

        # For demuxEM
        # The Dirichlet prior concentration parameter (alpha) on samples. An alpha value < 1.0 will make the prior sparse. [default: 0.0]
        Float? demuxEM_alpha_on_samples
        # Only demultiplex cells/nuclei with at least <demuxEM_min_num_umis> of UMIs. [default: 100]
        Int? demuxEM_min_num_umis
        # Any cell/nucleus with less than <count> hashtags from the signal will be marked as unknown. [default: 10.0]
        Float? demuxEM_min_signal_hashtag
        # The random seed used in the KMeans algorithm to separate empty ADT droplets from others. [default: 0]
        Int? demuxEM_random_state
        # Generate a series of diagnostic plots, including the background/signal between HTO counts, estimated background probabilities, HTO distributions of cells and non-cells etc. [default: true]
        Boolean demuxEM_generate_diagnostic_plots = true
        # Generate violin plots using gender-specific genes (e.g. Xist). <demuxEM_generate_gender_plot> is a comma-separated list of gene names
        String? demuxEM_generate_gender_plot
        # DemuxEM version
        String demuxEM_version = "0.1.7"
        # Number of CPUs used
        Int demuxEM_num_cpu = 8
        # Disk space in GB
        Int demuxEM_disk_space = 20
        # Memory size string for demuxEM
        String demuxEM_memory = "10G"

        # For souporcell
        # Number of expected clusters when doing clustering
        Int souporcell_num_clusters = 1
        # If true, run souporcell in de novo mode without reference genotypes; and if a reference genotype vcf file is provided in the sample sheet, use it only for matching the cluster labels computed by souporcell. If false, run souporcell with --known_genotypes option using the reference genotype vcf file specified in sample sheet, and souporcell_rename_donors is required in this case.
        Boolean souporcell_de_novo_mode = true
        # Users can provide a common variants list in VCF format for Souporcell to use, instead of calling SNPs de novo
        File? souporcell_common_variants
        # Skip remap step. Only recommended in non denovo mode or common variants are provided
        Boolean souporcell_skip_remap = false
        # A comma-separated list of donor names for renaming clusters achieved by souporcell
        String souporcell_rename_donors = ""
        # Souporcell version to use. Available versions: "2020.07", "2021.03"
        String souporcell_version = "2021.03"
        # Number of CPUs to request for souporcell per pair
        Int souporcell_num_cpu = 32
        # Disk space (integer) in GB needed for souporcell per pair
        Int souporcell_disk_space = 500
        # Memory size string for souporcell per pair
        String souporcell_memory = "120G"

        # For popscle (demuxlet/freemuxlet)
        # Minimum mapping quality to consider (lower MQ will be ignored) [default: 20]
        Int? popscle_min_MQ
        # Minimum distance to the tail (lower will be ignored) [default: 0]
        Int? popscle_min_TD
        # Tag representing readgroup or cell barcodes, in the case to partition the BAM file into multiple groups. For 10x genomics, use CB  [default: "CB"]
        String? popscle_tag_group
        # Tag representing UMIs. For 10x genomics, use UB  [default: "UB"]
        String? popscle_tag_UMI
        # Default is 0, means to use demuxlet, if this number > 0, use freemuxlet
        Int popscle_num_samples = 0
        # FORMAT field to extract the genotype, likelihood, or posterior from
        String popscle_field = "GT"
        # Grid of alpha to search for [default: "0.1,0.2,0.3,0.4,0.5"]
        String? popscle_alpha
        # Popscle version. Available versions: "2021.05", "0.1b"
        String popscle_version = "2021.05"
        # A comma-separated list of donor names for renaming clusters achieved by freemuxlet
        String? popscle_rename_donors
        # Number of CPUs used for popscle per pair
        Int popscle_num_cpu = 1
        # Memory size string for popscle per pair
        String popscle_memory = "120G"
        # Extra disk space (integer) in GB needed for popscle per pair
        Int popscle_extra_disk_space = 100

        # Version of config docker image to use. This docker is used for parsing the input sample sheet for downstream execution. Available options: "0.2", "0.1"
        String config_version = "0.2"
    }

    # Reference Index TSV
    String ref_index_file = if backend == "aws" then "s3://gred-cumulus-ref/resources/cellranger/index.tsv" else "gs://regev-lab/resources/cellranger/index.tsv"
    # Cumulus project's private repo.
    String cumulus_private_registry = if backend == "aws" then "752311211819.dkr.ecr.us-west-2.amazonaws.com" else "gcr.io/gred-cumulus-sb-01-991a49c4"
    # Google cloud zones, default to Roche Science Cloud zones
    String zones = "us-west1-a us-west1-b us-west1-c"

    call dmp.demultiplexing as demultiplexing {
        input:
            input_sample_sheet = input_sample_sheet,
            output_directory = output_directory,
            genome = genome,
            demultiplexing_algorithm = demultiplexing_algorithm,
            min_num_genes = min_num_genes,
            docker_registry = docker_registry,
            preemptible = preemptible,
            awsMaxRetries = awsMaxRetries,
            zones = zones,
            backend = backend,
            demuxEM_alpha_on_samples = demuxEM_alpha_on_samples,
            demuxEM_min_num_umis = demuxEM_min_num_umis,
            demuxEM_min_signal_hashtag = demuxEM_min_signal_hashtag,
            demuxEM_random_state = demuxEM_random_state,
            demuxEM_generate_diagnostic_plots = demuxEM_generate_diagnostic_plots,
            demuxEM_generate_gender_plot = demuxEM_generate_gender_plot,
            ref_index_file = ref_index_file,
            demuxEM_version = demuxEM_version,
            demuxEM_num_cpu = demuxEM_num_cpu,
            demuxEM_disk_space = demuxEM_disk_space,
            demuxEM_memory = demuxEM_memory,
            souporcell_num_clusters = souporcell_num_clusters,
            souporcell_de_novo_mode = souporcell_de_novo_mode,
            souporcell_common_variants = souporcell_common_variants,
            souporcell_skip_remap = souporcell_skip_remap,
            souporcell_rename_donors = souporcell_rename_donors,
            souporcell_version = souporcell_version,
            souporcell_num_cpu = souporcell_num_cpu,
            souporcell_disk_space = souporcell_disk_space,
            souporcell_memory = souporcell_memory,
            popscle_min_MQ = popscle_min_MQ,
            popscle_min_TD = popscle_min_TD,
            popscle_tag_group = popscle_tag_group,
            popscle_tag_UMI = popscle_tag_UMI,
            popscle_num_samples = popscle_num_samples,
            popscle_field = popscle_field,
            popscle_alpha = popscle_alpha,
            popscle_version = popscle_version,
            popscle_rename_donors = popscle_rename_donors,
            popscle_num_cpu = popscle_num_cpu,
            popscle_memory = popscle_memory,
            popscle_extra_disk_space = popscle_extra_disk_space,
            config_version = config_version
    }

    if (write_to_dsdb && defined(dsdb_id)) {
        call dsdb.write_to_DataSetDB_demux as write_to_dsdb_demux {
            input:
                input_zarr_files = demultiplexing.output_zarr_files,
                index_remap_csv = antibody_index_mapping_csv,
                dsdb_id = select_first([dsdb_id]),
                dsdb_project_authors = dsdb_project_authors,
                backend = backend,
                docker_registry = cumulus_private_registry,
                zones = zones,
                num_cpu = 4,
                memory = "10G",
                disk_space = 100,
                preemptible = preemptible,
                awsMaxRetries = awsMaxRetries
        }
    }

    output {
        Array[String] output_folders = demultiplexing.output_folders
        Array[File] output_zarr_files = demultiplexing.output_zarr_files
        String? dsdb_project_version = write_to_dsdb_demux.dsdb_project_version
    }
}
