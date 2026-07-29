#!/usr/bin/env bash
# Create the cheapest GCP GPU VM that Isaac Sim's own docs support: n1-standard-8 + one T4.
# Standard (on-demand) by default -- set SPOT=true for a cheaper, preemptible VM instead.
# Verify current GCP pricing yourself before a long run.
#
# Override any of these via environment variables, e.g.:
#   PROJECT_ID=my-proj ZONE=us-central1-b ./scripts/gcp_create_vm.sh
#   SPOT=true PROJECT_ID=my-proj ./scripts/gcp_create_vm.sh   # cheaper, can be stopped anytime
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project id}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-g1-baseline-t4}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR="${ACCELERATOR:-nvidia-tesla-t4}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-100}"
SPOT="${SPOT:-false}"

extra_args=()
if [[ "${SPOT}" == "true" ]]; then
  echo "Creating Spot VM '${INSTANCE_NAME}' (${MACHINE_TYPE} + ${ACCELERATOR}) in ${ZONE}..."
  echo "Spot VMs are cheaper but can be stopped by GCP at any time it needs the capacity back."
  extra_args+=(--provisioning-model=SPOT --instance-termination-action=STOP)
else
  echo "Creating standard (on-demand) VM '${INSTANCE_NAME}' (${MACHINE_TYPE} + ${ACCELERATOR}) in ${ZONE}..."
fi

gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=${ACCELERATOR},count=1" \
  --image-family="ubuntu-2204-lts" \
  --image-project="ubuntu-os-cloud" \
  --boot-disk-size="${BOOT_DISK_SIZE_GB}" \
  --boot-disk-type="pd-ssd" \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  "${extra_args[@]+"${extra_args[@]}"}"

echo
echo "Done. Next:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID}"
echo "  # then run scripts/remote_setup.sh on the VM"

if [[ "${SPOT}" == "true" ]]; then
  echo
  echo "If the VM gets Spot-preempted, it will STOP (not delete) -- restart with:"
  echo "  gcloud compute instances start ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID}"
fi
