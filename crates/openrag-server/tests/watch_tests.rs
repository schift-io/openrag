use std::collections::HashSet;
use std::time::Duration;

use openrag_server::watch::{plan, DiskFile, DocsWatchConfig, Step, WatchFolder, WatchedDoc, DEFAULT_TYPES};
use uuid::Uuid;

type R = Result<(), Box<dyn std::error::Error>>;

#[test]
fn folders_come_as_a_json_list_or_a_comma_list() -> R {
    let cfg = DocsWatchConfig::parse(
        r#"["/docs/", {"path": "/mnt/hr", "bucket": "hr", "types": ["PDF", ".hwp"]}]"#,
        None,
        None,
        None,
        None,
    )?
    .ok_or("no config")?;
    let defaults: Vec<String> = DEFAULT_TYPES.iter().map(ToString::to_string).collect();
    assert_eq!(
        cfg.folders,
        vec![
            WatchFolder { path: "/docs".into(), bucket: "docs".into(), types: defaults },
            WatchFolder { path: "/mnt/hr".into(), bucket: "hr".into(), types: vec!["pdf".into(), "hwp".into()] },
        ]
    );
    assert_eq!(cfg.interval, Duration::from_secs(30));

    let cfg = DocsWatchConfig::parse("/a, /b", Some("[\"md\", \"docx\"]"), Some("kb"), Some("5"), Some("acme"))?
        .ok_or("no config")?;
    assert_eq!(cfg.folders.len(), 2);
    assert!(cfg.folders.iter().all(|f| f.bucket == "kb" && f.types == ["md", "docx"]));
    assert_eq!(cfg.interval, Duration::from_secs(5));
    assert_eq!(cfg.org.as_deref(), Some("acme"));

    assert!(DocsWatchConfig::parse("[]", None, None, None, None)?.is_none());
    Ok(())
}

#[test]
fn bad_folder_settings_are_refused() {
    let err = |dirs: &str, types: Option<&str>, secs: Option<&str>| {
        DocsWatchConfig::parse(dirs, types, None, secs, None).err().map(|e| e.to_string()).unwrap_or_default()
    };
    assert!(err("docs", None, None).contains("absolute path"));
    assert!(err("/a,/a/", None, None).contains("twice"));
    assert!(err("/a", Some("pdf,xlsx"), None).contains("'xlsx' is not a file type"));
    assert!(err("/a", None, Some("0")).contains("at least 1"));
    assert!(err("[\"/a\"", None, None).contains("not a JSON list"));
}

fn doc(key: &str, sha: &str) -> WatchedDoc {
    WatchedDoc { id: Uuid::new_v4(), key: key.into(), sha: sha.into(), root: "/docs".into() }
}

fn file(key: &str, sha: Option<&str>) -> DiskFile {
    DiskFile { key: key.into(), sha: sha.map(Into::into), root: "/docs".into() }
}

fn none() -> HashSet<String> {
    HashSet::new()
}

#[test]
fn unchanged_new_changed_and_deleted_files() {
    let same = doc("/docs/a.md", "A");
    let edited = doc("/docs/b.md", "B");
    let gone = doc("/docs/c.md", "C");
    let files = [file("/docs/a.md", Some("A")), file("/docs/b.md", Some("B2")), file("/docs/new.pdf", Some("N"))];
    let steps = plan(&files, &[same.clone(), edited.clone(), gone.clone()], &none(), false);
    assert_eq!(
        steps,
        vec![
            Step::Keep { doc: same.id, file: 0 },
            Step::Ingest { file: 1, replaces: Some(edited.id) },
            Step::Ingest { file: 2, replaces: None },
            Step::Retire { doc: gone.id },
        ]
    );
}

#[test]
fn moved_and_renamed_files_keep_their_document() {
    let a = doc("/docs/규정/a.md", "A");
    let b = doc("/docs/b.md", "B");
    // a.md moved to another folder; b.md renamed and a new b.md written in its place.
    let files = [file("/docs/보관/a.md", Some("A")), file("/docs/b.md", Some("NEW")), file("/docs/b-old.md", Some("B"))];
    let steps = plan(&files, &[a.clone(), b.clone()], &none(), false);
    assert_eq!(
        steps,
        vec![
            Step::Move { doc: b.id, file: 2 },
            Step::Move { doc: a.id, file: 0 },
            Step::Ingest { file: 1, replaces: None },
        ]
    );
}

#[test]
fn swapped_names_are_two_moves_and_copies_need_nothing() {
    let x = doc("/docs/x.md", "X");
    let y = doc("/docs/y.md", "Y");
    let steps = plan(&[file("/docs/x.md", Some("Y")), file("/docs/y.md", Some("X"))], &[x.clone(), y.clone()], &none(), false);
    assert_eq!(steps, vec![Step::Move { doc: y.id, file: 0 }, Step::Move { doc: x.id, file: 1 }]);

    let steps = plan(&[file("/docs/x.md", Some("X")), file("/docs/x 복사본.md", Some("X"))], std::slice::from_ref(&x), &none(), false);
    assert_eq!(steps, vec![Step::Keep { doc: x.id, file: 0 }]);
    // The original deleted, the copy stays: the document moves to the copy.
    let steps = plan(&[file("/docs/x 복사본.md", Some("X"))], std::slice::from_ref(&x), &none(), false);
    assert_eq!(steps, vec![Step::Move { doc: x.id, file: 0 }]);
}

#[test]
fn files_being_written_and_missing_folders_keep_documents() {
    let a = doc("/docs/a.md", "A");
    let b = doc("/docs/b.md", "B");
    // a.md is still being written; b.md is gone, but a file in progress may be where it went.
    let steps = plan(&[file("/docs/a.md", None)], &[a.clone(), b.clone()], &none(), true);
    assert!(steps.is_empty(), "{steps:?}");
    // The folder holding b.md is unmounted.
    let unavailable: HashSet<String> = ["/docs".to_string()].into();
    let steps = plan(&[], &[a, b], &unavailable, false);
    assert!(steps.is_empty(), "{steps:?}");
}
