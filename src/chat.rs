//! Chat formatting with the Jinja template embedded in the GGUF
//! (`tokenizer.chat_template`), rendered by minijinja.

use anyhow::{Context, Result};
use minijinja::{Environment, Error, ErrorKind, context};

use crate::gguf::GgufFile;

/// Used when a model ships without a template.
const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
}

impl ChatTemplate {
    pub fn from_gguf(f: &GgufFile, tokens: &crate::tokenizer::Tokenizer) -> Result<Self> {
        let src = f.get_str("tokenizer.chat_template").unwrap_or(CHATML).to_owned();
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |_fmt: String| -> String { today() });
        env.add_template_owned("chat", src).context("parsing chat template")?;
        Ok(Self {
            env,
            bos: tokens.bos.map(|b| tokens.token_str(b).to_owned()).unwrap_or_default(),
            eos: tokens.eos.map(|e| tokens.token_str(e).to_owned()).unwrap_or_default(),
        })
    }

    pub fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let msgs: Vec<_> =
            messages.iter().map(|m| context! { role => m.role.clone(), content => m.content.clone() }).collect();
        let tmpl = self.env.get_template("chat")?;
        Ok(tmpl.render(context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos.clone(),
            eos_token => self.eos.clone(),
        })?)
    }

    /// True if the rendered prompt already begins with the BOS string, in
    /// which case the tokenizer must not add another one.
    pub fn starts_with_bos(&self, rendered: &str) -> bool {
        !self.bos.is_empty() && rendered.starts_with(&self.bos)
    }
}

/// Today's date as "24 Nov 2026" (what Llama 3 templates ask strftime for).
fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // Howard Hinnant's days-to-civil algorithm.
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    const NAMES: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!("{day:02} {} {year}", NAMES[(month - 1) as usize])
}
