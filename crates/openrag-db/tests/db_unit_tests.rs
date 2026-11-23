use openrag_db::{encode_text_array, encode_uuid_array, sha256_hex, EMBEDDED_MIGRATIONS};
use uuid::Uuid;

#[test]
fn embedded_migrations_are_ordered_and_clean() {
    assert_eq!(EMBEDDED_MIGRATIONS.len(), 14);
    let expected = [
        ("000", "000_init.sql"),
        ("001", "001_iam_arn.sql"),
        ("002", "002_vectors_corpus.sql"),
        ("003", "003_pii_capsule.sql"),
        ("004", "004_llm_wiki.sql"),
        ("005", "005_gateway_plugins.sql"),
        ("006", "006_eval_bench.sql"),
        ("007", "007_memory_redaction_indexing.sql"),
        ("008", "008_concepts_tags.sql"),
        ("009", "009_cli_auth_codes.sql"),
        ("010", "010_guard_sk_key_pattern.sql"),
        ("011", "011_vector_dim_256.sql"),
        ("012", "012_drop_page_figure_vectors.sql"),
        ("013", "013_llm_cli_providers.sql"),
    ];

    let excluded_db = ["schift", "db"].join("-");
    let excluded_ce = ["schift", "rag", "ce"].join("-");

    for (spec, (ver, fname)) in EMBEDDED_MIGRATIONS.iter().zip(expected.iter()) {
        assert_eq!(spec.version, *ver);
        assert_eq!(spec.filename, *fname);
        assert!(!spec.sql.trim().is_empty());
        assert!(!spec.sql.contains(&excluded_db));
        assert!(!spec.sql.contains(&excluded_ce));
    }

    let h1 = sha256_hex("SELECT 1;");
    let h2 = sha256_hex("SELECT 1;");
    assert_eq!(h1.len(), 64);
    assert_eq!(h1, h2);
}

#[test]
fn array_encoding_for_set_config_handles_empty_single_and_escaping() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(encode_uuid_array(&[]), "{}");
    let u1 = Uuid::parse_str("11111111-1111-1111-1111-111111111111")?;
    let u2 = Uuid::parse_str("22222222-2222-2222-2222-222222222222")?;
    assert_eq!(
        encode_uuid_array(&[u1]),
        "{11111111-1111-1111-1111-111111111111}"
    );
    assert_eq!(
        encode_uuid_array(&[u1, u2]),
        "{11111111-1111-1111-1111-111111111111,22222222-2222-2222-2222-222222222222}"
    );

    assert_eq!(encode_text_array(&[]), "{}");
    assert_eq!(
        encode_text_array(&["arn:openrag:acme:fin/*".to_string()]),
        "{\"arn:openrag:acme:fin/*\"}"
    );
    assert_eq!(
        encode_text_array(&["a,b".to_string(), "c\"d".to_string()]),
        "{\"a,b\",\"c\\\"d\"}"
    );
    Ok(())
}
