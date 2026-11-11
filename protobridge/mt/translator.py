from __future__ import annotations

from pathlib import Path
import re
import ctranslate2
from huggingface_hub import hf_hub_download, snapshot_download
from tokenizers import Tokenizer

NLLB_CT2_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
NLLB_TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"

ISO_TO_NLLB: dict[str, str] = {
    "es": "spa_Latn",
    "en": "eng_Latn",
    "pt": "por_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
}

GAMING_GLOSSARY_SUBS_ES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbile\s+titan(s)?\b", re.IGNORECASE), "titán de bilis"),
    (re.compile(r"\bstratagem(s)?\b", re.IGNORECASE), "estratagema"),
    (re.compile(r"\breinforce(ments)?\b", re.IGNORECASE), "refuerzos"),
    (re.compile(r"\bresupply\b", re.IGNORECASE), "reabastecimiento"),
    (re.compile(r"\beagle\s+500\s*kg\b", re.IGNORECASE), "águila de quinientos kilos"),
    (re.compile(r"\bhellbomb\b", re.IGNORECASE), "bomba infernal"),
    (re.compile(r"\bstalker(s)?\b", re.IGNORECASE), "acechador"),
    (re.compile(r"\bcharger(s)?\b", re.IGNORECASE), "cargador"),
)


class TacticalTranslatorEngine:
    """
    Traductor neuronal multilingüe local basado en NLLB-200-distilled-600M (CTranslate2 INT8).
    """

    def __init__(self, models_dir: Path) -> None:
        ct2_dir = models_dir / "nllb_200_ct2_int8"
        snapshot_download(
            repo_id=NLLB_CT2_REPO,
            local_dir=str(ct2_dir),
            local_dir_use_symlinks=False,
        )
        tok_file = ct2_dir / "tokenizer.json"
        if not tok_file.exists():
            hf_hub_download(
                repo_id=NLLB_TOKENIZER_REPO,
                filename="tokenizer.json",
                local_dir=str(ct2_dir),
                local_dir_use_symlinks=False,
            )

        self._tokenizer = Tokenizer.from_file(str(tok_file))
        self._translator = ctranslate2.Translator(
            str(ct2_dir),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=4,
        )
        self.translate("Ready.", src_lang="en", tgt_lang="es")

    def translate(self, text: str, src_lang: str = "en", tgt_lang: str = "es") -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        if src_lang == tgt_lang:
            return cleaned

        src_code = ISO_TO_NLLB.get(src_lang, "eng_Latn")
        tgt_code = ISO_TO_NLLB.get(tgt_lang, "spa_Latn")

        enc = self._tokenizer.encode(cleaned, add_special_tokens=False)
        source_tokens = [src_code] + list(enc.tokens) + ["</s>"]

        results = self._translator.translate_batch(
            [source_tokens],
            target_prefix=[[tgt_code]],
            beam_size=1,
            max_decoding_length=128,
        )
        out_tokens = results[0].hypotheses[0]
        skip_tags = set(ISO_TO_NLLB.values()) | {"</s>", "<s>", "<pad>"}
        token_ids: list[int] = []
        for tok in out_tokens:
            if tok in skip_tags:
                continue
            tid = self._tokenizer.token_to_id(tok)
            if tid is not None:
                token_ids.append(int(tid))

        translated = self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if tgt_lang == "es":
            for pattern, replacement in GAMING_GLOSSARY_SUBS_ES:
                translated = pattern.sub(replacement, translated)

        return translated

    def translate_en_to_es(self, english_text: str) -> str:
        return self.translate(english_text, src_lang="en", tgt_lang="es")
