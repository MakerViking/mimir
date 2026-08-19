//! TEMPORARY: prints ASTs to design binding/receiver extraction. Delete.
#[cfg(test)]
mod t {
    use crate::languages::Lang;
    use tree_sitter::Parser;

    fn sexp(lang: Lang, src: &str) {
        let mut p = Parser::new();
        p.set_language(&lang.language()).unwrap();
        let tree = p.parse(src, None).unwrap();
        println!("\n===== {:?} =====\n{}", lang, tree.root_node().to_sexp());
    }

    #[test]
    fn probe() {
        sexp(
            Lang::Java,
            r#"
class Repo {
  private Database db;
  void save(Item item) {
    Database store = new Database();
    store.put(item);
    this.db.put(item);
    Formatter.upper(item);
  }
}
"#,
        );
        sexp(
            Lang::CSharp,
            r#"
class Repo {
  private Database db;
  void Save(Item item) {
    Database store = new Database();
    store.Put(item);
    this.db.Put(item);
    Formatter.Upper(item);
  }
}
"#,
        );
        sexp(
            Lang::Cpp,
            r#"
class Repo {
  Database* db;
  void save(Item item) {
    Database store;
    store.put(item);
    this->db->put(item);
    Formatter::upper(item);
  }
};
"#,
        );
        sexp(
            Lang::Kotlin,
            r#"
class Repo(val db: Database) {
    fun save(item: Item) {
        val store = Database()
        store.put(item)
        db.put(item)
        Formatter.upper(item)
    }
}
"#,
        );
        sexp(
            Lang::Swift,
            r#"
class Repo {
    let db: Database
    func save(item: Item) {
        let store = Database()
        store.put(item)
        self.db.put(item)
        Formatter.upper(item)
    }
}
"#,
        );
        sexp(
            Lang::Php,
            r#"<?php
class Repo {
    private Database $db;
    public function save(Item $item) {
        $store = new Database();
        $store->put($item);
        $this->db->put($item);
        Formatter::upper($item);
    }
}
"#,
        );
    }
}
