use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::OpenRagError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessScope {
    Private,
    Dept,
    Group,
    Org,
}

impl AccessScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Private => "private",
            Self::Dept => "dept",
            Self::Group => "group",
            Self::Org => "org",
        }
    }
}

impl fmt::Display for AccessScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AccessScope {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "private" => Ok(Self::Private),
            "dept" => Ok(Self::Dept),
            "group" => Ok(Self::Group),
            "org" => Ok(Self::Org),
            other => Err(OpenRagError::Validation(format!(
                "invalid access_scope '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sensitivity {
    Share,
    Private,
    Zdr,
    Normal,
    Confidential,
    Secret,
}

impl Sensitivity {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Share => "share",
            Self::Private => "private",
            Self::Zdr => "zdr",
            Self::Normal => "normal",
            Self::Confidential => "confidential",
            Self::Secret => "secret",
        }
    }
}

impl fmt::Display for Sensitivity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct AuthLevel(u8);

impl AuthLevel {
    pub const MIN: Self = Self(1);
    pub const MAX: Self = Self(10);

    pub fn new(level: u8) -> Result<Self, OpenRagError> {
        if (1..=10).contains(&level) {
            Ok(Self(level))
        } else {
            Err(OpenRagError::Validation(format!(
                "auth_level must be within 1..=10, got {level}"
            )))
        }
    }

    pub const fn as_u8(self) -> u8 {
        self.0
    }

    pub const fn as_i16(self) -> i16 {
        self.0 as i16
    }
}

impl TryFrom<u8> for AuthLevel {
    type Error = OpenRagError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<i16> for AuthLevel {
    type Error = OpenRagError;

    fn try_from(value: i16) -> Result<Self, Self::Error> {
        if (1..=10).contains(&value) {
            Ok(Self(value as u8))
        } else {
            Err(OpenRagError::Validation(format!(
                "auth_level must be within 1..=10, got {value}"
            )))
        }
    }
}

impl From<AuthLevel> for u8 {
    fn from(value: AuthLevel) -> Self {
        value.0
    }
}

impl fmt::Display for AuthLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IamAction {
    Read,
    Write,
    Search,
    #[serde(rename = "pii:unmask")]
    PiiUnmask,
    Admin,
    #[serde(rename = "*")]
    Wildcard,
}

impl IamAction {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Search => "search",
            Self::PiiUnmask => "pii:unmask",
            Self::Admin => "admin",
            Self::Wildcard => "*",
        }
    }

    pub const fn matches(self, requested: Self) -> bool {
        matches!(self, Self::Wildcard) || matches!(requested, Self::Wildcard) || self as u8 == requested as u8
    }
}

impl fmt::Display for IamAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for IamAction {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "search" => Ok(Self::Search),
            "pii:unmask" => Ok(Self::PiiUnmask),
            "admin" => Ok(Self::Admin),
            "*" => Ok(Self::Wildcard),
            other => Err(OpenRagError::Validation(format!(
                "invalid iam action '{other}'"
            ))),
        }
    }
}
