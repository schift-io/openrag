use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PiiCategory {
    RrnKr,
    PhoneKr,
    Email,
    Passport,
    CreditCard,
    BankAccount,
    PersonName,
    AddressKr,
    SecretKey,
    CustomDict,
}

impl PiiCategory {
    pub fn parse_alias(raw: &str) -> Option<Self> {
        let norm = raw.trim().to_ascii_lowercase().replace('-', "_");
        match norm.as_str() {
            "rrn_kr" | "rrn" | "resident_registration_number" | "jumin" => Some(Self::RrnKr),
            "phone_kr" | "phone" | "mobile" | "tel" | "telephone" => Some(Self::PhoneKr),
            "email" | "e_mail" | "mail" => Some(Self::Email),
            "passport" | "passport_kr" => Some(Self::Passport),
            "credit_card" | "card" | "card_number" => Some(Self::CreditCard),
            "bank_account" | "bank" | "account_number" => Some(Self::BankAccount),
            "person_name" | "name" | "name_kr" | "person" => Some(Self::PersonName),
            "address_kr" | "address" | "addr" => Some(Self::AddressKr),
            "secret_key" | "api_key" | "secret" | "credential" => Some(Self::SecretKey),
            "custom_dict" | "custom" => Some(Self::CustomDict),
            _ => None,
        }
    }

    pub const fn as_canonical_str(self) -> &'static str {
        match self {
            Self::RrnKr => "rrn_kr",
            Self::PhoneKr => "phone_kr",
            Self::Email => "email",
            Self::Passport => "passport",
            Self::CreditCard => "credit_card",
            Self::BankAccount => "bank_account",
            Self::PersonName => "person_name",
            Self::AddressKr => "address_kr",
            Self::SecretKey => "secret_key",
            Self::CustomDict => "custom_dict",
        }
    }

    pub const fn token_tag(self) -> &'static str {
        match self {
            Self::RrnKr => "RRN",
            Self::PhoneKr => "PHONE",
            Self::Email => "EMAIL",
            Self::Passport => "PASSPORT",
            Self::CreditCard => "CARD",
            Self::BankAccount => "BANK",
            Self::PersonName => "NAME",
            Self::AddressKr => "ADDRESS",
            Self::SecretKey => "SECRET",
            Self::CustomDict => "CUSTOM",
        }
    }
}
