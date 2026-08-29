#!/bin/bash
# Deploy the cleaned red-parcel checkpoint to the Orin.
# Run on the LAPTOP (git bash).  $1 = best step (e.g. 020000)
# Never overwrites an existing bundle; prints checkpoint identity per handoff
# appendix A (profile name, absolute paths, prompt, dataset root, SHA256).
set -eu
BEST="${1:-020000}"
R='$HOME/parcel_smolvla/20260828_red_parcel_clean_50'
TAG=red_clean
N=/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean
OLD=/home/nvidia/work/telop/models/smolvla_onearm_20k_20260805

LAST=$(ssh -o BatchMode=yes a800new "ls $R/outputs/train_$TAG/checkpoints/ | grep -E '^[0-9]{6}\$' | sort | tail -1")
echo "== best=$BEST last=$LAST -> $N =="

send() {  # $1 step  $2 dest
  ssh -o BatchMode=yes a800new "cd $R/outputs/train_$TAG/checkpoints/$1 && tar -cf - pretrained_model" \
  | ssh -o BatchMode=yes armstrong-orin "mkdir -p $N && cd $N && rm -rf .t && mkdir .t && tar -xf - -C .t && rm -rf $2 && mv .t/pretrained_model $2 && rmdir .t"
  A=$(ssh -o BatchMode=yes a800new "cd $R/outputs/train_$TAG/checkpoints/$1/pretrained_model && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")
  B=$(ssh -o BatchMode=yes armstrong-orin "cd $N/$2 && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")
  echo "   $2  a800new=$A  orin=$B"
  [ "$A" = "$B" ] || { echo "CHECKSUM MISMATCH ($2)"; exit 1; }
}

send "$BEST" checkpoint
send "$LAST" checkpoint_last

# manifests, cleaning report, validation reports
ssh -o BatchMode=yes a800new "cd $R && tar -cf - split_manifest.json manifest validation_reports 2>/dev/null" \
  | ssh -o BatchMode=yes armstrong-orin "tar -xf - -C $N/ 2>/dev/null || true"

ssh -o BatchMode=yes armstrong-orin "
set -eu
N=$N; OLD=$OLD
[ -d \$N/vlm ] || cp -al \$OLD/vlm \$N/vlm 2>/dev/null || cp -r \$OLD/vlm \$N/vlm
[ -d \$N/deployment ] || cp -al \$OLD/deployment \$N/deployment 2>/dev/null || cp -r \$OLD/deployment \$N/deployment
sed 's|smolvla_onearm_20k_20260805|smolvla_20260828_red_parcel_clean|g' \$OLD/start_policy_server.sh > \$N/start_policy_server.sh
chmod +x \$N/start_policy_server.sh
python3 - <<'PY'
import json
base = '$N'
VLM = base + '/vlm'
for ck in ['checkpoint', 'checkpoint_last']:
    p = f'{base}/{ck}/config.json'
    c = json.load(open(p)); c['vlm_model_name'] = VLM; c['load_vlm_weights'] = True
    json.dump(c, open(p, 'w'), indent=2)
    q = f'{base}/{ck}/policy_preprocessor.json'
    d = json.load(open(q)); n = [0]
    def walk(o):
        if isinstance(o, dict):
            for k, v in list(o.items()):
                if k == 'tokenizer_name' and isinstance(v, str) and v != VLM:
                    o[k] = VLM; n[0] += 1
                else: walk(v)
        elif isinstance(o, list):
            for x in o: walk(x)
    walk(d); json.dump(d, open(q, 'w'), indent=2)
    print(ck, 'tokenizer localised:', n[0])
PY
cd \$N && find . -type f -not -name SHA256SUMS | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
echo '--- bundle identity (handoff appendix A) ---'
echo \"profile   : red_parcel_clean\"
echo \"bundle    : \$N\"
echo \"checkpoint: \$N/checkpoint\"
echo \"prompt    : 把箱子里的红色包裹拿出来放到桌子上。\"
echo \"ckpt sha  : \$(cd \$N/checkpoint && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-32)\"
ls \$N; du -sh \$N"
echo "== deploy done =="
