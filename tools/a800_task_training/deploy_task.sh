#!/bin/bash
# Deploy one trained task to the Orin. Run on the LAPTOP (git bash).
#   $1 = run_dir_name on a800new
#   $2 = train_tag
#   $3 = best step (e.g. 015000)
#   $4 = bundle name on Orin (e.g. smolvla_20260827_red_parcel_out_table)
# Streams best + last checkpoints a800new -> laptop -> Orin, verifies SHA,
# assembles the bundle (vlm/deployment hard-linked from the old bundle) and
# localises the tokenizer path (steps list!) for fully offline loading.
set -eu
RUNDIR=$1; TAG=$2; BEST=$3; BUNDLE=$4
R="\$HOME/parcel_smolvla/$RUNDIR"
N="/home/nvidia/work/telop/models/$BUNDLE"
OLD="/home/nvidia/work/telop/models/smolvla_onearm_20k_20260805"

LAST=$(ssh -o BatchMode=yes a800new "ls $R/outputs/train_$TAG/checkpoints/ | grep -E '^[0-9]{6}\$' | sort | tail -1")
echo "== best=$BEST last=$LAST -> $BUNDLE =="

send() {  # $1 step  $2 dest subdir
  ssh -o BatchMode=yes a800new "cd $R/outputs/train_$TAG/checkpoints/$1 && tar -cf - pretrained_model" \
  | ssh -o BatchMode=yes armstrong-orin "mkdir -p $N && rm -rf $N/.tmp_$2 && mkdir -p $N/.tmp_$2 && tar -xf - -C $N/.tmp_$2 && rm -rf $N/$2 && mv $N/.tmp_$2/pretrained_model $N/$2 && rmdir $N/.tmp_$2"
  A=$(ssh -o BatchMode=yes a800new "cd $R/outputs/train_$TAG/checkpoints/$1/pretrained_model && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")
  B=$(ssh -o BatchMode=yes armstrong-orin "cd $N/$2 && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")
  echo "   $2: a800new=$A orin=$B"
  [ "$A" = "$B" ] || { echo "CHECKSUM MISMATCH for $2"; exit 1; }
}

send "$BEST" checkpoint
send "$LAST" checkpoint_last

# manifests / reports
ssh -o BatchMode=yes a800new "cd $R && tar -cf - split_manifest.json preprocessing_report.json validation_reports 2>/dev/null" \
  | ssh -o BatchMode=yes armstrong-orin "tar -xf - -C $N/ 2>/dev/null || true"

ssh -o BatchMode=yes armstrong-orin "
set -eu
[ -d $N/vlm ] || cp -al $OLD/vlm $N/vlm 2>/dev/null || cp -r $OLD/vlm $N/vlm
[ -d $N/deployment ] || cp -al $OLD/deployment $N/deployment 2>/dev/null || cp -r $OLD/deployment $N/deployment
sed 's|smolvla_onearm_20k_20260805|$BUNDLE|g' $OLD/start_policy_server.sh > $N/start_policy_server.sh
chmod +x $N/start_policy_server.sh
python3 - <<PY
import json
VLM = '$N/vlm'
for ck in ['checkpoint', 'checkpoint_last']:
    p = f'$N/{ck}/config.json'
    c = json.load(open(p))
    c['vlm_model_name'] = VLM
    c['load_vlm_weights'] = True
    json.dump(c, open(p, 'w'), indent=2)
    q = f'$N/{ck}/policy_preprocessor.json'
    d = json.load(open(q)); n = 0
    def walk(o):
        global n
        if isinstance(o, dict):
            for k, v in list(o.items()):
                if k == 'tokenizer_name' and isinstance(v, str) and v != VLM:
                    o[k] = VLM; n += 1
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(d)
    json.dump(d, open(q, 'w'), indent=2)
    print(ck, 'vlm+tokenizer localised, tokenizer fields changed:', n)
PY
cd $N && find . -type f -not -name SHA256SUMS | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
echo BUNDLE_READY; ls $N; du -sh $N"
echo "== deploy done: $BUNDLE (best=$BEST last=$LAST) =="
